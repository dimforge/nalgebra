tmp=_git_distcheck
nalgebra_lib_path=lib
nalgebra_doc_path=doc
all:
	mkdir -p $(nalgebra_lib_path)
	rustc src/nalgebra.rs --out-dir $(nalgebra_lib_path) --opt-level 3

test:
	mkdir -p $(nalgebra_lib_path)
	rustc --test src/nalgebra.rs --opt-level 3 -o test~ && ./test~
	rm test~

bench:
	rustc --test src/nalgebra.rs --opt-level 3 -o bench~ && ./bench~ --bench
	rm bench~

doc:
	mkdir -p $(nalgebra_doc_path)
	rustdoc src/nalgebra.rs

distcheck:
	rm -rf $(tmp)
	git clone --recursive . $(tmp)
	make -C $(tmp)
	make -C $(tmp) test
	rm -rf $(tmp)

.PHONY:doc
.PHONY:test
