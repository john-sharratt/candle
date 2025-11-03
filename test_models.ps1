$prompt = "The quick brown fox jumps over the lazy dog. This is a test of a much longer prompt to see how fast we can process input tokens. We want to measure the throughput of the model when given a large amount of text to process at once. This will help us understand if there are any bottlenecks in the input processing path. Let me add more text here to make this prompt even longer and more representative of real-world usage where users might paste in large documents or have long conversations. The model should be able to handle this efficiently with the GPU acceleration we have enabled. More text here to reach a good token count for testing purposes. Adding even more content to ensure we have a substantial number of tokens to process."

Write-Host "Testing Qwen25_1B_Q2..." -ForegroundColor Cyan
.\target\release\examples\quantized-qwen2-instruct.exe --which qwen25-1b-q2 --prompt $prompt --sample-len 50 --temperature 0

Write-Host "`nTesting Qwen25_1B_Q3..." -ForegroundColor Cyan
.\target\release\examples\quantized-qwen2-instruct.exe --which qwen25-1b-q3 --prompt $prompt --sample-len 50 --temperature 0

Write-Host "`nTesting Qwen25_7B_Q5..." -ForegroundColor Cyan  
.\target\release\examples\quantized-qwen2-instruct.exe --which qwen25-7b-q5 --prompt $prompt --sample-len 50 --temperature 0

Write-Host "`nTesting Qwen25_7B_F16..." -ForegroundColor Cyan
.\target\release\examples\quantized-qwen2-instruct.exe --which qwen25-7b-f16 --prompt $prompt --sample-len 50 --temperature 0

Write-Host "`nTesting Qwen30_8B_Q5..." -ForegroundColor Cyan
.\target\release\examples\quantized-qwen2-instruct.exe --which qwen30-8b-q5 --prompt $prompt --sample-len 50 --temperature 0

Write-Host "`nTesting Qwen30_8B_F16..." -ForegroundColor Cyan
.\target\release\examples\quantized-qwen2-instruct.exe --which qwen30-8b-f16 --prompt $prompt --sample-len 50 --temperature 0

Write-Host "`nTesting Qwen25_14B_Q5..." -ForegroundColor Cyan
.\target\release\examples\quantized-qwen2-instruct.exe --which qwen25-14b-q5 --prompt $prompt --sample-len 50 --temperature 0

Write-Host "`nTesting Qwen25_14B_Q6..." -ForegroundColor Cyan
.\target\release\examples\quantized-qwen2-instruct.exe --which qwen25-14b-q6 --prompt $prompt --sample-len 50 --temperature 0
