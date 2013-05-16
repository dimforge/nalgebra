all:
	rust build src/nalgebra.rc --out-dir lib # rustpkg install

doc:
	rust doc src/nalgebra.rc --output-dir doc

.PHONY:doc
