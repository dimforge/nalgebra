all:
	CARGO_INCREMENTAL=1 cargo build --features "arbitrary serde-serialize"

doc:
	CARGO_INCREMENTAL=1 cargo doc --no-deps --features "arbitrary serde-serialize"

bench:
	cargo bench

test:
	cargo test --features "arbitrary serde-serialize"
