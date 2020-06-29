all:
	cargo test --features "debug arbitrary serde-serialize abomonation-serialize compare"
	# cargo check --features "debug arbitrary serde-serialize"

doc:
	cargo doc --no-deps --features "debug arbitrary serde-serialize abomonation"

bench:
	cargo bench

test:
	cargo test --features "debug arbitrary serde-serialize abomonation-serialize compare"
