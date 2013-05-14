all:
	rustpkg install

doc:
	rust doc src/nalgebra.rc --output-dir doc

.PHONY:doc
