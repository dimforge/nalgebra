## What is this
This folder contains all the matrices used for `nalgebra-sparse` benchmark.

The matrices are listed here. For more details, you can find at [link1](https://sparse.tamu.edu/Williams/), [link2](http://dx.doi.org/10.1016/j.parco.2008.12.006).

|Name |File Name   | Hard Drive Size (approximately)|
|-|-|-|
|[Protein](https://sparse.tamu.edu/Williams/pdb1HYS)|pdb1HYS|55M|
|[FEM/Spheres](https://sparse.tamu.edu/Williams/consph)|consph|83M|
|[FEM/Cantilever](https://sparse.tamu.edu/Williams/cant)|cant|57M|
|[Wind Tunnel](https://sparse.tamu.edu/Boeing/pwtk)|pwtk|155M|
|[FEM/Harbor](https://sparse.tamu.edu/Bova/rma10)|rma10|59M|
|[QCD](https://sparse.tamu.edu/QCD/conf5_4-8x8-05)|conf5_4-8x8-05|89M|
|[FEM/Ship](https://sparse.tamu.edu/DNVS/shipsec1)|shipsec1|83M|
|[Economics](https://sparse.tamu.edu/Williams/mac_econ_fwd500)|mac_econ_fwd500|32M|
|[Epidemiology](https://sparse.tamu.edu/Williams/mc2depi)|mc2depi|37M|
|[FEM/Accelerator](https://sparse.tamu.edu/Williams/cop20k_A)|cop20k_A|27M|
|[Circuit](https://sparse.tamu.edu/Hamm/scircuit)|scircuit|28M|
|[webbase](https://sparse.tamu.edu/Williams/webbase-1M)|webbase-1M|68M|
|[LP](https://sparse.tamu.edu/Mittelmann/rail4284)|rail4284|149M|

## How to use it
Simply run `python data.py`. It will automatically download and unzip all the matrices needed.

Note: python3 and Internet connection are required. No additional dependency required.

If you want to remove the matrices, simply run `python data.py clean`.