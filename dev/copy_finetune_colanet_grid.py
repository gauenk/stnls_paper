
import cache_io

def main():


    name0 = ".cache_io/finetune_colanet_grid_i2lab"
    version0 = "v1"
    name1 = ".cache_io/finetune_colanet_grid"
    version1 = "v1"
    exps = cache_io.get_exps("exps/finetune_colanet_grid.cfg")
    cache_io.copy.enames(name0,version0,
                         name1,version1,exps,overwrite=True)

if __name__ == "__main__":
    main()
