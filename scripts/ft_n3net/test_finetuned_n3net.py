"""

Test the finetuned n3net to show the differences

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- data mangling --
import numpy as np

# -- network configs --
from icml23 import test_model
from icml23 import reports

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    cfg_file = "exps/test_finetuned_n3net.cfg"
    records = cache_io.run_exps(cfg_file,test_model.run,skip_loop=True)

    # -- print table information --
    lnames = {"048a":"FT(wt=3)","f5ae":"FT(wt=0)","pret":"Orig."}
    records['pretrained_path'] = records['pretrained_path'].str.slice(0,4)
    records['strred'] = records['strred']*100.
    ppath = records['pretrained_path'].to_numpy()
    labels = np.zeros(len(records)).astype(str)
    for label,name in lnames.items():
        labels = np.where(ppath==label,name,labels)
    records['labels'] = labels
    gfields = ['labels','sigma','wt']
    fmts = ["%2.2f","%.3f","%2.2f"]
    afields = ['psnrs','ssims','strred']
    agg = lambda x: np.mean(np.stack(x))
    records = records.groupby(gfields).agg({k:agg for k in afields})
    for g,gdf in records.groupby(gfields):
        i_fxn = lambda f,fmt: fmt % (gdf[f].to_numpy()[0])
        print(list(g),"/".join([i_fxn(f,fmt) for f,fmt in zip(afields,fmts)]))

if __name__ == "__main__":
    main()
