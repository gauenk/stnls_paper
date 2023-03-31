This is the repo to manage all the experiments for our paper.

Our system is uses Ubuntu 20.02 OS with two TITAN RTX GPUs with an Intel i7 Processor. We use Python 3.10.0 and CUDA 11.4.

1.) To install the package, run the following commands:

`./scripts/download_repos.sh` # skip this for ICML submission

`python -m pip install --upgrade pip`

`python -m pip install -r reqs/reqs_cu113.txt`

`python -m pip install -r reqs/reqs_local.txt`

2.) Installing the SWIG-wrapped Video Non-Local Bayes for their implementation of TV-L1 optical flow:

`cd ../svnlb/`

`./install.sh`

--- Networks --

Each network is contained within its pacakge name: COLA-Net is in colanet, LIDIA is in lidia, and N3Net is in n3net. You must setup the pretrained_root parameter to the absolute directory inside of each experimental configuration ("./exps") you want to run the experiment.

-- Data Hub, Cache IO, Dev Basics ---

To standardize dataset access, read/write experimental results, and share some common functions, we use homemade packages named data_hub, cache_io, and dev_basics. Simply installing the above libraries will ensure our code works properly. To re-run the results, the DAVIS and Set8 datasets must be installed (see Datasets).

-- Datasets ---

Please install the Set8 and DAVIS datasets into your system. The "data_hub" package requires you specify the paths in the "path.py" files under the "data_hub/lib/data_hub/sets/davis" and "data_hub/lib/data_hub/sets/set8" directory.

--- DNLS ---

Our primary code is contained within the DNLS module. The CUDA code for the DNLS module is within the "lib/csrc/search" directory. The additional cuda kernels are in the "tile", "tile_k" and "reducers" directory.

--- Benchmarking Note ---

When benchmarking our methods, we ensure nothing else on our computer is running. While we appreciate many different factors contribute to the final wall-clock time and memory consumption reported in our paper, we claim the order of magnitude repored in the paper is properly controlled.

--- Misc ---

Each visual element in the paper has a corresponding script. See "docs/visual_elements.txt" for a mapping.

Happy Hacking!
