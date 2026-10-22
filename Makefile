.PHONY: clean-ptx clean test

# The kernel cache is candle-kernels' precompiled/ archives and staged/ objects
# (candle-kernels/README.md); touching build.rs makes cargo run it again,
# which finds both gone and compiles every kernel.
clean-ptx:
	rm -rf candle-kernels/precompiled candle-kernels/staged
	touch candle-kernels/build.rs
	touch candle-examples/build.rs
	touch candle-flash-attn/build.rs

clean:
	cargo clean

test:
	cargo test

all: test
