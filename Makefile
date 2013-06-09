nalgebra_lib_path=lib
nalgebra_doc_path=doc
all:
	mkdir -p $(nalgebra_lib_path)
	rust build src/nalgebra.rc --out-dir $(nalgebra_lib_path)

test:
	mkdir -p $(nalgebra_lib_path)
	rust test src/nalgebra.rc
	rm nalgebratest~

doc:
	mkdir -p $(nalgebra_doc_path)
	rust doc src/nalgebra.rc --output-dir $(nalgebra_doc_path)

.PHONY:doc
.PHONY:test
