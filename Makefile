nalgebra_lib_path=lib
all:
	rust build src/nalgebra.rc --out-dir $(nalgebra_lib_path)

test:
	rust test src/nalgebra.rc
	rm nalgebratest~

doc:
	rust test src/nalgebra.rc

.PHONY:doc, test
