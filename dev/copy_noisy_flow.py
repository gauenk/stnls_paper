
import cache_io

def main():

    name0 = ".cache_io"
    version0 = "noisy_flow"
    name1 = ".cache_io/noisy_flow"
    version1 = "v1"
    exps = cache_io.get_exps("exps/noisy_flow.cfg")
    cache_io.copy.enames(name0,version0,
                         name1,version1,exps)

if __name__ == "__main__":
    main()
