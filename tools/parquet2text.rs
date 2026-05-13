/// Standalone tool to convert WikiText parquet to raw text
/// Usage: cargo run --manifest-path tools/Cargo.toml --bin parquet2text -- input.parquet output.txt
use std::env;
use std::fs::File;
use std::io::Write;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input.parquet> <output.txt>", args[0]);
        std::process::exit(1);
    }
    let input = &args[1];
    let output = &args[2];

    let file = File::open(input).expect("Failed to open parquet file");
    let reader = parquet::file::reader::SerializedFileReader::new(file)
        .expect("Failed to create parquet reader");

    use parquet::file::reader::FileReader;
    let mut out = File::create(output).expect("Failed to create output file");
    let mut row_count = 0;

    let iter = reader.get_row_iter(None).expect("Failed to get row iterator");
    for row in iter {
        let row = row.expect("Failed to read row");
        // WikiText parquet has a single "text" column
        for (name, field) in row.get_column_iter() {
            if name == "text" {
                if let parquet::record::Field::Str(s) = field {
                    writeln!(out, "{}", s).expect("Failed to write");
                    row_count += 1;
                }
            }
        }
    }
    eprintln!("Wrote {} lines to {}", row_count, output);
}
