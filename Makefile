tmp=_git_distcheck
nalgebra_lib_path=lib
nalgebra_doc_path=doc
all:
	mkdir -p $(nalgebra_lib_path)
	rust build src/lib.rs --out-dir $(nalgebra_lib_path) --opt-level 2

test:
	mkdir -p $(nalgebra_lib_path)
	rust test src/lib.rs
	rm libtest~

bench:
	rustc --test src/lib.rs --opt-level 2 -o bench~ && ./bench~ --bench
	rm bench~

doc:
	mkdir -p $(nalgebra_doc_path)
	rust doc src/lib.rs --output-dir $(nalgebra_doc_path)

distcheck:
	rm -rf $(tmp)
	git clone --recursive . $(tmp)
	make -C $(tmp)
	make -C $(tmp) test
	rm -rf $(tmp)

.PHONY:doc
.PHONY:test
