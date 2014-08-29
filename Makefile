tmp=_git_distcheck

all:
	cargo build --release

test:
	cargo test

bench:
	rustc --test src/lib.rs --opt-level 3 -o bench~ && ./bench~ --bench
	rm bench~

doc:
	cargo doc

clean:
	cargo clean

distcheck:
	rm -rf $(tmp)
	git clone --recursive . $(tmp)
	make -C $(tmp)
	make -C $(tmp) test
	rm -rf $(tmp)

.PHONY:doc
.PHONY:test
