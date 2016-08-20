tmp=_git_distcheck

all:
	cargo build --release --features "arbitrary generic_sizes abstract_algebra"

test:
	cargo test --features "arbitrary generic_sizes abstract_algebra"


bench:
	cargo bench --features "arbitrary generic_sizes abstract_algebra"


doc:
	cargo doc --no-deps --features "arbitrary generic_sizes abstract_algebra"


clean:
	cargo clean

distcheck:
	rm -rf $(tmp)
	git clone --recursive . $(tmp)
	make -C $(tmp)
	make -C $(tmp) test
	make -C $(tmp) bench
	rm -rf $(tmp)

.PHONY:doc
.PHONY:test
