import cache_io

def main():

    name0 = ".cache_io"
    version0 = "modulate_gpu_runtime"
    name1 = ".cache_io/modulate_gpu_runtime"
    version1 = "v1"
    exps = cache_io.get_exps("exps/modulate_gpu_runtime.cfg")
    cache_io.copy.enames(name0,version0,
                         name1,version1,exps)

if __name__ == "__main__":
    main()
