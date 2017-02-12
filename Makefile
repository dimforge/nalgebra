all:
	CARGO_INCREMENTAL=1 cargo build --features "arbitrary"

doc:
	CARGO_INCREMENTAL=1 cargo doc

bench:
	cargo bench

test:
	CARGO_INCREMENTAL=1 cargo test --features "arbitrary"
