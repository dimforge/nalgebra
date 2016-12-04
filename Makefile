all:
	cargo build --features "arbitrary"

doc:
	cargo doc

bench:
	cargo bench

test:
	cargo test --features "arbitrary"
