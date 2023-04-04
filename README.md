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

--- STNLS ---

Our primary code is contained within the STNLS module. The CUDA code for the STNLS module is within the "lib/csrc/search" directory. The additional cuda kernels are in the "tile", "tile_k" and "reducers" directory.

--- Benchmarking Note ---

When benchmarking our methods, we ensure nothing else on our computer is running. While we appreciate many different factors contribute to the final wall-clock time and memory consumption reported in our paper, we claim the order of magnitude repored in the paper is properly controlled.

--- Misc ---

Each visual element in the paper has a corresponding script. See "docs/visual_elements.txt" for a mapping.

Happy Hacking!


# Running a "trte_*" script

## Training

1.) Init the uuids in the base cache:
    `python ./scripts/trte_NAME/train.py --skip_loop`
2.) Launch training on separate machines:
    `sbatch_py ./scripts/trte_NAME/train.py NUM_EXPS EXPS_PER_PROC -U -J TRAIN_NAME_HERE`
3.) Optionally, merge the results
    `merge_cache DEST_CACHE NUM_EXPS EXPS_PER_PROC TRAIN_NAME_HERE`

## Testing

1.) Init the pretrained_paths in the base cache configs:
    `python ./scripts/trte_NAME/test.py --skip_loop`
2.) Launch training on separate machines:
    `sbatch_py ./scripts/trte_NAME/test.py NUM_EXPS EXPS_PER_PROC -U -J TEST_NAME_HERE`
3.) Non-optionally, merge the results. Many ways to do this:
    `python ./scripts/trte_NAME/test.py --job_id TEST_NAME_HERE --nexps NUM_EXPS --nexps_pp EXPS_PER_PROC --merge_cache --fast --skip_loop`
    `python ./scripts/trte_NAME/test.py NUM_EXPS EXPS_PER_PROC -J TEST_NAME_HERE --merge_cache --fast --skip_loop`
    `merge_cache DEST_CACHE NUM_EXPS EXPS_PER_PROC TEST_NAME_HERE`
4.) Create a pickle to store the final results for faster io each time.
    `python ./scripts/f2f/test.py --skip_loop`
