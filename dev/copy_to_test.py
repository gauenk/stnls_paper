"""

Copy the lidia test results to the main test results

"""

# -- dev basics --
from dev_basics.reports import deno_report
from dev_basics.utils import vid_io

# -- caching results --
import cache_io

def copy_lidia(dest,ow=False):

    # -- info --
    print("Copying LIDIA")

    # -- source cache --
    wt_l = [3,0]
    cdirs = [".cache_io/test_lidia_yesT",".cache_io/test_lidia_noT_r3"]
    # cdirs = [".cache_io/test_nets_v9"]
    cache_name = "v1"
    # wt_l = [0]
    # cdirs = [".cache_io/test_lidia_noT_r3"]
    exps = cache_io.get_exps("exps/test_lidia.cfg")
    for wt,cache_dir in zip(wt_l,cdirs):

        # -- open --
        src = cache_io.ExpCache(cache_dir,cache_name)

        # -- get exps --
        exps_f = [e for e in exps if e.wt == wt]

        # -- copy to main --
        # cache_io.copy.exp_cache(src,dest,exps_f,overwrite=ow)

    # -- check final --
    # exps = [e for e in exps if e.wt == 3]
    df = dest.to_records(exps)
    print(len(df))
    # fields = ["arch_name","wt","sigma","dname"]
    # for g,gdf in df.groupby(fields):
    #     print(g,num_empty(gdf))
    # print(df['deno_fns'])
    # print(len(dest.to_records(exps)))

def num_empty(df):
    nempty = 0
    for _,row in df.iterrows():
        fns = row['deno_fns'][0]
        if fns[0] == "":
            nempty += 1
    return nempty

def copy_colanet(dest,ow=False):

    # -- info --
    print("Copying COLA-Net")

    # -- source cache --
    cache_dir = ".cache_io/test_colanet_01_04"
    cache_name = "v1"
    src = cache_io.ExpCache(cache_dir,cache_name)

    # -- get exps --
    exps = cache_io.get_exps("exps/test_colanet.cfg")
    # print(len(exps))
    # print(len(src.to_records(exps)))

    # -- copy to main --
    # cache_io.copy.exp_cache(src,dest,exps,overwrite=ow)

    # -- check --
    print(len(dest.to_records(exps)))

def copy_n3net(dest,ow=False):

    # -- info --
    print("Copying N3Net")

    # -- source cache --
    cache_dir = ".cache_io/test_nets_v2"
    cache_name = "v1"
    src = cache_io.ExpCache(cache_dir,cache_name)

    # -- get exps --
    exps = cache_io.get_exps("exps/test_n3net.cfg")

    # -- copy to main --
    # cache_io.copy.exp_cache(src,dest,exps,overwrite=ow)

    # -- check --
    print(len(dest.to_records(exps)))

def main():

    # -- destinaton cache --
    cache_dir = ".cache_io/test_nets_v11"
    cache_name = "v1"
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- copying --
    copy_lidia(cache,True)
    copy_colanet(cache,True)
    copy_n3net(cache,True)


if __name__ == "__main__":
    main()
