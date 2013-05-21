nalgebra_lib_path=lib
nalgebra_doc_path=doc
all:
	rust build src/nalgebra.rc --out-dir $(nalgebra_lib_path)

test:
	rust test src/nalgebra.rc
	rm nalgebratest~

doc:
	rust doc src/nalgebra.rc --output-dir $(nalgebra_doc_path)

.PHONY:doc
.PHONY:test
