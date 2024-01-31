"""

Benchmark different nets

"""
# -- sys --
import os

# -- caching results --
import cache_io

# -- bench --
from dev_basics.trte import bench

def run_bench(net_cfg):
    # -- read config --
    bench_fn = "exps/bench/%s_bench.cfg" % net_cfg.arch_name
    exps = cache_io.exps.load(bench_fn)
    assert len(exps) == 1
    cfg = exps[0]

    # -- update shared data fields --
    fields = ["nframes","isize","cropmode",
              "dname","tr_set","batch_size","arch_name"]
    for field in fields:
        cfg[field] = net_cfg[field]
    cfg['sigma'] = 30.
    cfg['python_module'] = cfg['arch_name']
    cfg['bench_fwd_only'] = True

    # -- time & mem --
    res = bench.run(cfg)

    # -- params --
    if net_cfg.arch_name != "vnlb":
        vshape = (1,cfg.nframes,cfg.dd_in,256,256)
        model = bench.load_model(cfg).to("cuda")
        model.forward = bench.wrap_flows(model,vshape)
        summ = bench.th_summary(model, input_size=vshape, verbose=0)
        res.total_params = summ.total_params / 1e6
        res.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        res.trainable_params = res.trainable_params/1e6
    else:
        res.total_params = 0
        res.trainable_params = 0
    return res

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    def clear_fxn(num,cfg): return False
    exps = cache_io.exps.load("exps/bench/compare_net_fwd.cfg")
    print("Num Exps: ",len(exps))

    results = cache_io.run_exps(exps,run_bench,preset_uuids=False,
                                name=".cache_io/bench/compare_net_fwd",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/bench/compare_net_fwd.pkl",
                                records_reload=True,use_wandb=False,
                                proj_name="compare_net_fwd")

    # -- view --
    if len(results) == 0: return
    print(results.head())
    fields = ['arch_name','timer_fwd_nograd','res_fwd_nograd',
              'trainable_params']
    print(results[fields])

if __name__ == "__main__":
    main()
