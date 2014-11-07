tmp=_git_distcheck

all:
	cargo build --release

test:
	cargo test

bench:
	cargo bench

doc:
	cargo doc

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
