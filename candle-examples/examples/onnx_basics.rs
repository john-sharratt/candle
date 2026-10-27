//! ONNX model inspection and evaluation via `candle-onnx`.
//!
//! Three subcommands over an `.onnx` protobuf file: `print --file <p>` dumps the
//! parsed `ModelProto` and every graph node, `ops --file <p>` lists the op types
//! the graph uses with their counts and its declared inputs and outputs — what to
//! check against the interpreter before running a new export, without printing
//! its weights — and `simple-eval --file <p>` builds zero-filled inputs from the
//! graph's declared input shapes (skipping names that are graph initializers) and
//! runs the ONNX interpreter over them. Requires the `onnx` feature.

use anyhow::Result;
use candle::{Device, Tensor};

use clap::{Parser, Subcommand};

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Print {
        #[arg(long)]
        file: String,
    },
    Ops {
        #[arg(long)]
        file: String,
    },
    SimpleEval {
        #[arg(long)]
        file: String,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Print { file } => {
            let model = candle_onnx::read_file(file)?;
            println!("{model:?}");
            let graph = model.graph.unwrap();
            for node in graph.node.iter() {
                println!("{node:?}");
            }
        }
        Command::Ops { file } => {
            let model = candle_onnx::read_file(file)?;
            println!(
                "opset {:?}",
                model
                    .opset_import
                    .iter()
                    .map(|o| (o.domain.as_str(), o.version))
                    .collect::<Vec<_>>()
            );
            let graph = model.graph.unwrap();
            let weights: std::collections::HashSet<_> =
                graph.initializer.iter().map(|i| i.name.as_str()).collect();
            for input in graph
                .input
                .iter()
                .filter(|i| !weights.contains(i.name.as_str()))
            {
                println!("input {}: {:?}", input.name, input.r#type);
            }
            for output in graph.output.iter() {
                println!("output {}: {:?}", output.name, output.r#type);
            }
            let mut ops = std::collections::BTreeMap::<&str, usize>::new();
            for node in graph.node.iter() {
                *ops.entry(node.op_type.as_str()).or_default() += 1;
            }
            println!(
                "{} initializers, {} nodes",
                graph.initializer.len(),
                graph.node.len()
            );
            for (op, n) in ops {
                println!("{n:6}  {op}");
            }
        }
        Command::SimpleEval { file } => {
            let model = candle_onnx::read_file(file)?;
            let graph = model.graph.as_ref().unwrap();
            let constants: std::collections::HashSet<_> =
                graph.initializer.iter().map(|i| i.name.as_str()).collect();
            let mut inputs = std::collections::HashMap::new();
            for input in graph.input.iter() {
                use candle_onnx::onnx::tensor_proto::DataType;
                if constants.contains(input.name.as_str()) {
                    continue;
                }

                let type_ = input.r#type.as_ref().expect("no type for input");
                let type_ = type_.value.as_ref().expect("no type.value for input");
                let value = match type_ {
                    candle_onnx::onnx::type_proto::Value::TensorType(tt) => {
                        let dt = match DataType::try_from(tt.elem_type) {
                            Ok(dt) => match candle_onnx::dtype(dt) {
                                Some(dt) => dt,
                                None => {
                                    anyhow::bail!(
                                        "unsupported 'value' data-type {dt:?} for {}",
                                        input.name
                                    )
                                }
                            },
                            type_ => anyhow::bail!("unsupported input type {type_:?}"),
                        };
                        let shape = tt.shape.as_ref().expect("no tensortype.shape for input");
                        let dims = shape
                                .dim
                                .iter()
                                .map(|dim| match dim.value.as_ref().expect("no dim value") {
                                    candle_onnx::onnx::tensor_shape_proto::dimension::Value::DimValue(v) => Ok(*v as usize),
                                    candle_onnx::onnx::tensor_shape_proto::dimension::Value::DimParam(_) => Ok(42),
                                })
                                .collect::<Result<Vec<usize>>>()?;
                        Tensor::zeros(dims, dt, &Device::Cpu)?
                    }
                    type_ => anyhow::bail!("unsupported input type {type_:?}"),
                };
                println!("input {}: {value:?}", input.name);
                inputs.insert(input.name.clone(), value);
            }
            let outputs = candle_onnx::simple_eval(&model, inputs)?;
            for (name, value) in outputs.iter() {
                println!("output {name}: {value:?}")
            }
        }
    }
    Ok(())
}
