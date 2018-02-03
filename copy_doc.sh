#! /bin/sh

out_dir=./docs/rustdoc
nalgebra_dir=../nalgebra-dev

echo "Generating the documentation..."
cd $nalgebra_dir; cargo doc --no-deps --features "arbitrary serde-serialize abomonation"
cd -
rm -rf docs/rustdoc
cp -r $nalgebra_dir/target/doc $out_dir

echo "... documentation generated!"

./fix_rustdoc.sh
