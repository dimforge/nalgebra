tmp=_git_distcheck
nalgebra_doc_path=doc

all:
	cargo build --release

test:
	cargo test

bench:
	rustc --test src/lib.rs --opt-level 3 -o bench~ && ./bench~ --bench
	rm bench~

doc:
	mkdir -p $(nalgebra_doc_path)
	rustdoc src/lib.rs

distcheck:
	rm -rf $(tmp)
	git clone --recursive . $(tmp)
	make -C $(tmp)
	make -C $(tmp) test
	rm -rf $(tmp)

.PHONY:doc
.PHONY:test
