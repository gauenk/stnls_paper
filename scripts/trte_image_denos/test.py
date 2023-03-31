
# -- misc --
import os
import numpy as np

# -- caching results --
import cache_io

# -- network configs --
from icml23 import test_model
from icml23 import reports

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- run/load exps --
    # exp_files = ["exps/test_n3net.cfg","exps/test_lidia.cfg","exps/test_colanet.cfg"]
    exp_files = ["exps/test_colanet.cfg"]
    exps = cache_io.get_exps(exp_files)
    print(len(exps))
    # exp_grid = np.arange(38*3,38*4).tolist()
    exp_grid = np.arange(8*4,8*5).tolist()
    exps = [exps[i] for i in exp_grid]
    print(len(exps))
    records = cache_io.run_exps(exps,test_model.run,
                                # name=".cache_io/test_models",
                                name=".cache_io/test_models_colanet",
                                version="v1",skip_loop=False,
                                # records_fn=".cache_io_pkl/test_models.pkl",
                                records_fn=".cache_io_pkl/test_models_colanet.pkl",
                                records_reload=False)

    # -- table report --
    reports.deno_table_v2.run(records)

    # -- latex reports --

    fields = ['arch_name','dname','sigma',
                      'vid_name','ws','wt',"bw"]
    fields_summ = ['arch_name','dname','ws','wt',"bw"]
    res_fields = ['psnrs','ssims','timer_deno','mem_alloc','mem_res']
    res_fmt = ['%2.3f','%1.3f','%2.3f','%2.3f','%2.3f','%2.3f']

    df_set8 = records[records['dname'] == "set8"]
    report = reports.deno_table_v2.run_latex(df_set8,fields,fields_summ,
                                          res_fields,res_fmt)
    print(report)

    df_davis = records[records['dname'] == "davis"]
    report = reports.deno_table_v2.run_latex(df_davis,fields,fields_summ,
                                          res_fields,res_fmt)
    print(report)

if __name__ == "__main__":
    main()
