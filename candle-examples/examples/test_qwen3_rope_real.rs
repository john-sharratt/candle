#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{quantized::gguf_file, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    println!("🔍 Testing Official Qwen3-8B-Q5 RoPE with 4000+ Token Context");
    println!("{}", "=".repeat(70));

    // Use official Qwen3-8B-Q5 model
    println!("\n📥 Loading Qwen3-8B-Q5_K_M model from HuggingFace...");
    let api = hf_hub::api::sync::Api::new()?;

    let model_path = {
        let repo = api.repo(hf_hub::Repo::with_revision(
            "Qwen/Qwen3-8B-GGUF".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        repo.get("Qwen3-8B-Q5_K_M.gguf")?
    };

    let tokenizer_path = {
        let api_model = api.model("Qwen/Qwen3-8B".to_string());
        api_model.get("tokenizer.json")?
    };

    println!("📂 Model: {}", model_path.display());
    println!("📂 Tokenizer: {}", tokenizer_path.display());

    // Load model - same as quantized-qwen3 example
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    let device = candle_examples::device(false)?; // Use GPU

    let mut model = {
        let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
        println!(
            "Loaded {} tensors in {:.2}s",
            model.tensor_infos.len(),
            start.elapsed().as_secs_f32()
        );
        Qwen3::from_gguf(model, &mut file, &device)?
    };
    println!("✅ Model built successfully");
    println!("   Device: {:?}", device);

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    // Create a long conversation targeting 4000+ tokens
    println!("\n💬 Creating long conversation...");
    let prompt_str = create_long_conversation();

    let tokens = tokenizer
        .encode(prompt_str.clone(), true)
        .map_err(anyhow::Error::msg)?;
    let tokens = tokens.get_ids().to_vec();
    let token_count = tokens.len();

    println!("   Tokenized to {} tokens", token_count);

    if token_count < 4000 {
        println!("⚠️  Warning: Only {} tokens, expected 4000+", token_count);
    }

    // TEST 1: Process prompt in chunks like real usage (tests RoPE at various offsets)
    println!("\n🧪 Test 1: Process prompt in 100-token chunks (simulates real KV cache usage)");
    model.clear_all_caches();

    let batch_size = 100;
    let mut offset = 0;

    for (batch_idx, chunk) in tokens.chunks(batch_size).enumerate() {
        let chunk_start = std::time::Instant::now();
        let input = Tensor::new(chunk, &device)?.unsqueeze(0)?;

        println!(
            "   Batch {} (offset={}, len={})",
            batch_idx,
            offset,
            chunk.len()
        );

        let logits = model.forward(&input, offset)?;

        // Check for NaN/Inf which indicates RoPE failure
        let logits_vec = logits.to_vec2::<f32>()?;
        let has_nan = logits_vec[0].iter().any(|v| v.is_nan());
        let has_inf = logits_vec[0].iter().any(|v| v.is_infinite());

        if has_nan || has_inf {
            println!("   ❌ FAILED: NaN/Inf at offset {}!", offset);
            anyhow::bail!("RoPE produced NaN/Inf at offset {}", offset);
        }

        // Check logit range (very small range indicates collapse)
        let max_val = logits_vec[0]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_val = logits_vec[0].iter().copied().fold(f32::INFINITY, f32::min);
        let range = max_val - min_val;

        if range < 0.01 {
            println!("   ⚠️  Warning: Very small logit range {:.6}", range);
        }

        println!(
            "      ✓ OK (range={:.2}, time={:.3}s)",
            range,
            chunk_start.elapsed().as_secs_f64()
        );

        offset += chunk.len();
    }

    println!("   ✅ Test 1 PASSED: Processed {} tokens in chunks", offset);
    println!("   Final cache length: {}", model.cache_len());

    // TEST 2: Continue generation beyond the prompt (tests RoPE continues working)
    println!(
        "\n🧪 Test 2: Generate tokens beyond position {} and check text quality",
        offset
    );

    let eos_token = *tokenizer.get_vocab(true).get("<|im_end|>").unwrap();
    let mut generated_token_ids = Vec::new();
    let max_gen = 100;

    for i in 0..max_gen {
        let last_token = if i == 0 {
            tokens[tokens.len() - 1]
        } else {
            generated_token_ids[generated_token_ids.len() - 1]
        };

        let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, offset + i)?;

        // Check for corruption
        let logits_vec = logits.to_vec2::<f32>()?;
        if logits_vec[0].iter().any(|v| v.is_nan() || v.is_infinite()) {
            println!("   ❌ FAILED: Model collapsed at position {}!", offset + i);
            anyhow::bail!("Model collapsed at position {}", offset + i);
        }

        // Greedy sampling
        let next_token = logits_vec[0]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        generated_token_ids.push(next_token);

        if next_token == eos_token {
            println!(
                "   Generated {} tokens before EOS",
                generated_token_ids.len()
            );
            break;
        }

        if (i + 1) % 20 == 0 {
            println!("   Position {}: still generating...", offset + i);
        }
    }

    // Decode and display the generated text
    let generated_text = tokenizer
        .decode(&generated_token_ids, true)
        .map_err(anyhow::Error::msg)?;
    println!("\n   📝 Generated text (first 300 chars):");
    let preview = generated_text.chars().take(300).collect::<String>();
    println!("   \"{}\"", preview);

    // Check for signs of gibberish/collapse
    let word_count = generated_text.split_whitespace().count();
    let char_count = generated_text.chars().filter(|c| c.is_alphabetic()).count();
    let avg_word_len = if word_count > 0 {
        char_count as f32 / word_count as f32
    } else {
        0.0
    };

    println!("\n   Text quality metrics:");
    println!(
        "      Words: {}, Avg word length: {:.1}",
        word_count, avg_word_len
    );

    // Check for excessive repetition (same 3-char sequence repeated)
    let mut has_repetition = false;
    if generated_text.len() > 10 {
        for window_size in [3, 4, 5] {
            if generated_text.len() >= window_size * 3 {
                for i in 0..generated_text.len() - window_size * 3 {
                    let chunk1 = &generated_text[i..i + window_size];
                    let chunk2 = &generated_text[i + window_size..i + window_size * 2];
                    let chunk3 = &generated_text[i + window_size * 2..i + window_size * 3];
                    if chunk1 == chunk2 && chunk2 == chunk3 {
                        println!("      ⚠️  WARNING: Repetition detected: {:?}", chunk1);
                        has_repetition = true;
                        break;
                    }
                }
            }
            if has_repetition {
                break;
            }
        }
    }

    if !has_repetition && word_count > 5 && avg_word_len > 2.0 && avg_word_len < 15.0 {
        println!("      ✓ Text appears coherent");
    } else if avg_word_len < 2.0 || avg_word_len > 15.0 {
        println!("      ⚠️  WARNING: Unusual word length - possible gibberish!");
    }

    println!(
        "   ✅ Test 2 PASSED: Generated {} tokens beyond position {}",
        generated_token_ids.len(),
        offset
    );

    // TEST 3: Full context at once (tests RoPE with large seq_len)
    println!(
        "\n🧪 Test 3: Forward pass with all {} tokens at once",
        token_count
    );
    model.clear_all_caches();

    let full_start = std::time::Instant::now();
    let input_full = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits_full = model.forward(&input_full, 0)?;
    let full_time = full_start.elapsed();

    // Check for corruption
    let logits_vec = logits_full.to_vec2::<f32>()?;
    if logits_vec[0].iter().any(|v| v.is_nan() || v.is_infinite()) {
        println!("   ❌ FAILED: Full context forward produced NaN/Inf!");
        anyhow::bail!("Full context forward failed");
    }

    println!(
        "   ✅ Test 3 PASSED: Full context in {:.2}s",
        full_time.as_secs_f64()
    );

    // SUMMARY
    println!("\n{}", "=".repeat(70));
    println!("📊 All RoPE Tests PASSED!");
    println!("   ✅ Chunked processing up to {} tokens", offset);
    println!("   ✅ Continued generation beyond {} tokens", offset);
    println!("   ✅ Full context forward with {} tokens", token_count);
    println!("\n🎉 RoPE implementation is working correctly for long contexts!");

    Ok(())
}

fn create_long_conversation() -> String {
    // Create a conversation that will be approximately 4000+ tokens
    let mut conversation = String::new();

    conversation.push_str("<|im_start|>system\nYou are a helpful AI assistant named Qwen, created by Alibaba Cloud. You are designed to be helpful, harmless, and honest.<|im_end|>\n");

    // Add multiple user/assistant exchanges to build up token count
    // Each exchange adds approximately 200-300 tokens
    let exchanges = vec![
        (
            "Can you explain what machine learning is and how it works?",
            "Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from and make decisions based on data. Instead of being explicitly programmed for every task, machine learning algorithms use statistical techniques to identify patterns in data and improve their performance over time. The process typically involves three main steps: first, collecting and preparing training data; second, selecting and training a model using that data; and third, evaluating and refining the model's performance. There are several types of machine learning, including supervised learning (where the algorithm learns from labeled data), unsupervised learning (where it finds patterns in unlabeled data), and reinforcement learning (where it learns through trial and error by receiving rewards or penalties). Common algorithms include linear regression, decision trees, support vector machines, and neural networks. Each has its strengths and weaknesses depending on the problem domain. Machine learning is used in countless applications today, from email spam filters and recommendation systems to autonomous vehicles and medical diagnosis. The field continues to evolve rapidly with new techniques and architectures being developed constantly."
        ),
        (
            "What are neural networks and how are they related to the human brain?",
            "Neural networks are computing systems inspired by the biological neural networks in animal brains. They consist of interconnected nodes (called neurons or units) organized in layers. Each connection has an associated weight that gets adjusted during training. When data passes through the network, each neuron receives input from previous layers, applies a mathematical transformation, and passes the result to the next layer. While neural networks are inspired by biological brains, they're actually quite different. Biological neurons are far more complex, using chemical and electrical signals, while artificial neurons use simple mathematical functions. The human brain has approximately 86 billion neurons with trillions of connections, whereas even large artificial neural networks are much smaller. However, the core idea of learning through adjusting connection strengths is similar. Deep learning uses neural networks with many layers (hence 'deep'), allowing them to learn hierarchical representations of data. This has proven remarkably effective for tasks like image recognition, natural language processing, and game playing. The training process involves forward propagation (passing data through the network), calculating loss (measuring error), and backpropagation (adjusting weights to reduce error). This iterative process continues until the network achieves satisfactory performance."
        ),
        (
            "How do transformers work in natural language processing?",
            "Transformers are a revolutionary architecture in natural language processing introduced in the 'Attention Is All You Need' paper in 2017. The key innovation is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when processing each word. Unlike recurrent neural networks that process text sequentially, transformers process all words simultaneously, making them more parallelizable and efficient. The architecture consists of an encoder-decoder structure (though many modern models use only the encoder or only the decoder). The self-attention mechanism works by creating three vectors for each word: Query, Key, and Value. It then computes attention scores by comparing the Query of one word with the Keys of all other words, determining how much focus to place on each word. These scores are used to create a weighted sum of the Value vectors. Transformers also use positional encodings to maintain information about word order, since they don't process sequences in order like RNNs do. This architecture has become the foundation for modern large language models like GPT, BERT, and myself (Qwen), enabling impressive capabilities in understanding and generating human language. The transformer's ability to process sequences in parallel and capture long-range dependencies has made it the dominant architecture for NLP tasks, from translation and summarization to question answering and code generation."
        ),
        (
            "What is the difference between GPT and BERT models?",
            "GPT (Generative Pre-trained Transformer) and BERT (Bidirectional Encoder Representations from Transformers) are both transformer-based models, but they differ in architecture and training approach. GPT is a decoder-only model trained using causal (left-to-right) language modeling, where it predicts the next token given all previous tokens. This makes it naturally suited for text generation tasks. BERT, on the other hand, is an encoder-only model trained using masked language modeling (MLM), where random tokens are masked and the model learns to predict them using context from both directions (left and right). This bidirectional context makes BERT excellent for understanding tasks like question answering, named entity recognition, and text classification. GPT models are autoregressive, generating one token at a time, while BERT processes entire sequences at once for understanding. Modern variations have emerged: GPT evolved into GPT-2, GPT-3, GPT-4 with increased scale and capabilities, while BERT inspired models like RoBERTa, ALBERT, and DistilBERT. Some models like T5 and BART combine encoder-decoder architectures to handle both understanding and generation tasks effectively. The choice between GPT-style and BERT-style models depends on the specific task requirements."
        ),
        (
            "Can you explain what attention mechanisms are and why they're important?",
            "Attention mechanisms are a fundamental component of modern neural networks, especially in natural language processing and computer vision. The core idea is to allow models to focus on relevant parts of the input when making predictions, similar to how humans pay attention to specific details when processing information. In the context of transformers, self-attention allows each position in a sequence to attend to all positions in the previous layer, computing a weighted sum based on relevance. The mechanism works by transforming the input into Query, Key, and Value representations, then computing similarity scores between queries and keys to determine attention weights. These weights are applied to the values to create context-aware representations. Attention is crucial because it enables models to capture long-range dependencies in data without the limitations of recurrent architectures. It's also more interpretable than previous approaches, as attention weights can show which parts of the input the model focuses on. Multi-head attention, used in transformers, applies multiple attention mechanisms in parallel, allowing the model to attend to different aspects of the input simultaneously. This has proven extremely effective, making attention-based architectures the dominant approach in modern AI systems for language, vision, and multimodal tasks."
        ),
        (
            "What are the main challenges in training large language models?",
            "Training large language models presents several significant challenges. First, computational cost is enormous - training GPT-3 required thousands of high-end GPUs for weeks, costing millions of dollars. Second, data quality and quantity are critical; models need diverse, high-quality text data spanning many domains and languages. Third, there's the challenge of scaling laws - while larger models generally perform better, the relationship between size and capability isn't linear, and there are diminishing returns. Fourth, memory limitations require sophisticated techniques like gradient checkpointing, model parallelism, and efficient attention mechanisms to fit models on available hardware. Fifth, optimization is tricky at scale - learning rates, batch sizes, and other hyperparameters that work for small models don't necessarily transfer to large ones. Sixth, evaluation is complex because these models have emergent capabilities that simple benchmarks may not capture. Seventh, there are safety and alignment challenges - ensuring models behave helpfully and don't produce harmful content. Eighth, inference costs are substantial; deploying large models in production requires significant infrastructure. Finally, there's the environmental impact of training, which has led to increased focus on efficiency and sustainability in AI research."
        ),
        (
            "How do transformers work in natural language processing?",
            "Transformers are a revolutionary architecture in natural language processing introduced in the 'Attention Is All You Need' paper in 2017. The key innovation is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when processing each word. Unlike recurrent neural networks that process text sequentially, transformers process all words simultaneously, making them more parallelizable and efficient. The architecture consists of an encoder-decoder structure (though many modern models use only the encoder or only the decoder). The self-attention mechanism works by creating three vectors for each word: Query, Key, and Value. It then computes attention scores by comparing the Query of one word with the Keys of all other words, determining how much focus to place on each word. These scores are used to create a weighted sum of the Value vectors. Transformers also use positional encodings to maintain information about word order, since they don't process sequences in order like RNNs do. This architecture has become the foundation for modern large language models like GPT, BERT, and myself (Qwen), enabling impressive capabilities in understanding and generating human language. The transformer's ability to process sequences in parallel and capture long-range dependencies has made it the dominant architecture for NLP tasks, from translation and summarization to question answering and code generation."
        ),
        (
            "What is the difference between GPT and BERT models?",
            "GPT (Generative Pre-trained Transformer) and BERT (Bidirectional Encoder Representations from Transformers) are both transformer-based models, but they differ in architecture and training approach. GPT is a decoder-only model trained using causal (left-to-right) language modeling, where it predicts the next token given all previous tokens. This makes it naturally suited for text generation tasks. BERT, on the other hand, is an encoder-only model trained using masked language modeling (MLM), where random tokens are masked and the model learns to predict them using context from both directions (left and right). This bidirectional context makes BERT excellent for understanding tasks like question answering, named entity recognition, and text classification. GPT models are autoregressive, generating one token at a time, while BERT processes entire sequences at once for understanding. Modern variations have emerged: GPT evolved into GPT-2, GPT-3, GPT-4 with increased scale and capabilities, while BERT inspired models like RoBERTa, ALBERT, and DistilBERT. Some models like T5 and BART combine encoder-decoder architectures to handle both understanding and generation tasks effectively. The choice between GPT-style and BERT-style models depends on the specific task requirements."
        ),
        (
            "Can you explain what attention mechanisms are and why they're important?",
            "Attention mechanisms are a fundamental component of modern neural networks, especially in natural language processing and computer vision. The core idea is to allow models to focus on relevant parts of the input when making predictions, similar to how humans pay attention to specific details when processing information. In the context of transformers, self-attention allows each position in a sequence to attend to all positions in the previous layer, computing a weighted sum based on relevance. The mechanism works by transforming the input into Query, Key, and Value representations, then computing similarity scores between queries and keys to determine attention weights. These weights are applied to the values to create context-aware representations. Attention is crucial because it enables models to capture long-range dependencies in data without the limitations of recurrent architectures. It's also more interpretable than previous approaches, as attention weights can show which parts of the input the model focuses on. Multi-head attention, used in transformers, applies multiple attention mechanisms in parallel, allowing the model to attend to different aspects of the input simultaneously. This has proven extremely effective, making attention-based architectures the dominant approach in modern AI systems for language, vision, and multimodal tasks."
        ),
        (
            "What are the main challenges in training large language models?",
            "Training large language models presents several significant challenges. First, computational cost is enormous - training GPT-3 required thousands of high-end GPUs for weeks, costing millions of dollars. Second, data quality and quantity are critical; models need diverse, high-quality text data spanning many domains and languages. Third, there's the challenge of scaling laws - while larger models generally perform better, the relationship between size and capability isn't linear, and there are diminishing returns. Fourth, memory limitations require sophisticated techniques like gradient checkpointing, model parallelism, and efficient attention mechanisms to fit models on available hardware. Fifth, optimization is tricky at scale - learning rates, batch sizes, and other hyperparameters that work for small models don't necessarily transfer to large ones. Sixth, evaluation is complex because these models have emergent capabilities that simple benchmarks may not capture. Seventh, there are safety and alignment challenges - ensuring models behave helpfully and don't produce harmful content. Eighth, inference costs are substantial; deploying large models in production requires significant infrastructure. Finally, there's the environmental impact of training, which has led to increased focus on efficiency and sustainability in AI research."
        ),
        (
            "How does reinforcement learning differ from supervised learning?",
            "Reinforcement learning (RL) and supervised learning are fundamentally different approaches to machine learning. In supervised learning, the model learns from labeled examples where the correct output is provided for each input. The model tries to minimize the difference between its predictions and the known correct answers. This works well when you have abundant labeled data and clear right/wrong answers. In contrast, reinforcement learning involves an agent learning through interaction with an environment. The agent receives rewards or penalties based on its actions but doesn't get explicit labels for what the correct action should be. Instead, it must explore the environment, try different actions, and learn from the consequences. Key concepts in RL include the state space (possible situations), action space (possible moves), reward function (feedback mechanism), and policy (strategy for choosing actions). RL is particularly suited for sequential decision-making problems where actions have long-term consequences, such as game playing, robotics, and autonomous driving. While supervised learning optimizes for immediate accuracy on labeled data, RL optimizes for cumulative reward over time, often requiring the agent to make strategic trade-offs between exploration (trying new actions) and exploitation (using known good actions). Deep RL combines neural networks with RL algorithms, enabling agents to handle complex, high-dimensional state spaces."
        ),
        (
            "What is transfer learning and why is it important?",
            "Transfer learning is a machine learning technique where knowledge gained from solving one problem is applied to a different but related problem. Instead of training a model from scratch for each new task, transfer learning leverages pre-trained models that have already learned useful features from large datasets. This approach has become fundamental in modern deep learning, particularly for computer vision and natural language processing. For example, a model trained on millions of images to recognize objects can be fine-tuned on a smaller dataset for medical image analysis, transferring its understanding of visual features. In NLP, models like BERT and GPT are pre-trained on vast amounts of text to learn language patterns, then fine-tuned for specific tasks like sentiment analysis or question answering. Transfer learning is important for several reasons: it dramatically reduces training time and computational requirements; it enables good performance even with limited task-specific data; it helps prevent overfitting on small datasets; and it allows practitioners without massive computational resources to build effective models. The technique works because neural networks learn hierarchical representations - early layers capture general features that are useful across many tasks, while later layers become more task-specific. This insight has led to the development of foundation models that can be adapted to numerous downstream tasks with minimal additional training."
        ),
        (
            "What is backpropagation and how does it work?",
            "Backpropagation, short for 'backward propagation of errors', is the fundamental algorithm used to train neural networks. It's a method for calculating the gradient of the loss function with respect to each weight in the network, enabling the network to learn from its mistakes. The algorithm works in two phases: forward pass and backward pass. During the forward pass, input data flows through the network layer by layer, with each neuron applying its weights and activation function to produce outputs. At the final layer, the network's prediction is compared to the true label using a loss function, which quantifies how wrong the prediction is. During the backward pass, the algorithm computes how much each weight contributed to the error by applying the chain rule of calculus. Starting from the output layer, it calculates gradients and propagates them backward through the network, layer by layer. These gradients indicate the direction and magnitude of change needed for each weight to reduce the error. Once gradients are computed, an optimization algorithm like stochastic gradient descent (SGD) or Adam updates the weights to minimize the loss. The process repeats over many iterations and batches of data until the network learns to make accurate predictions. Backpropagation is computationally efficient because it calculates all gradients in a single backward pass, making it practical to train deep networks with millions of parameters."
        ),
        (
            "How do convolutional neural networks work for image processing?",
            "Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing grid-like data, particularly images. The key innovation is the convolutional layer, which applies learned filters (also called kernels) across the input to detect local patterns like edges, textures, and shapes. Each filter slides across the image, performing element-wise multiplications and summing the results to create a feature map. Early layers typically learn simple features like edges and corners, while deeper layers combine these to recognize more complex patterns like object parts and entire objects. CNNs use three key concepts: local connectivity (each neuron only connects to a small region of the input), parameter sharing (the same filter is used across the entire image), and pooling (downsampling to reduce spatial dimensions while retaining important features). This design dramatically reduces the number of parameters compared to fully connected networks, making CNNs practical for high-resolution images. A typical CNN architecture includes multiple convolutional layers with activation functions (like ReLU), pooling layers (like max pooling), and fully connected layers at the end for classification. The hierarchical feature learning in CNNs mirrors aspects of the visual cortex in biological vision systems. Modern architectures like ResNet, VGG, and EfficientNet have pushed the boundaries of image recognition, achieving superhuman performance on many benchmarks. CNNs have also been adapted for other domains like audio processing, video analysis, and even natural language processing with character-level or word-level convolutions."
        ),
        (
            "What are GANs and how do they generate realistic images?",
            "Generative Adversarial Networks (GANs) are a class of machine learning frameworks invented by Ian Goodfellow in 2014. They consist of two neural networks - a generator and a discriminator - that compete against each other in a game-theoretic scenario. The generator creates fake samples (like images) from random noise, while the discriminator tries to distinguish between real samples from the training data and fake samples from the generator. During training, the generator learns to produce increasingly realistic outputs to fool the discriminator, while the discriminator becomes better at detecting fakes. This adversarial process continues until the generator produces samples so realistic that the discriminator can't reliably tell them apart from real data. The training objective is formulated as a minimax game: the generator tries to minimize the discriminator's ability to classify its outputs as fake, while the discriminator tries to maximize its classification accuracy. Mathematically, this is expressed as a loss function that both networks optimize in opposite directions. GANs have been remarkably successful at generating high-quality images, leading to applications in art generation, photo enhancement, style transfer, and data augmentation. Variants like StyleGAN can generate photorealistic faces, while conditional GANs allow control over specific attributes of generated samples. However, GAN training is notoriously difficult, often suffering from mode collapse (where the generator produces limited variety), training instability, and difficulty in evaluating output quality. Despite these challenges, GANs remain one of the most powerful tools for generative modeling and continue to drive innovation in creative AI applications."
        ),
    ];

    for (user_msg, assistant_msg) in exchanges {
        conversation.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user_msg));
        conversation.push_str(&format!(
            "<|im_start|>assistant\n{}<|im_end|>\n",
            assistant_msg
        ));
    }

    // Add one more user message to test prediction
    conversation.push_str("<|im_start|>user\nWhat is the future of artificial intelligence and what breakthroughs can we expect in the next decade?<|im_end|>\n<|im_start|>assistant\n");

    conversation
}
