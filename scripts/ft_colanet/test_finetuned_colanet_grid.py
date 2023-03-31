"""

Test the finetuned colanet models

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
    cfg_file = "exps/test_finetuned_colanet_grid.cfg"
    def clear_fxn(num,cfg):
        return False
        # return cfg.pretrained_path[:4] == "aa54"
        # return cfg.pretrained_path[:4] == "2b04"
        # return cfg.pretrained_path[:4] == "64de"
    records = cache_io.run_exps(cfg_file,test_model.run,
                                clear_fxn=clear_fxn,
                                skip_loop=False)

    # -- print table information --
    # lnames = {"aa54":"FT(wt=3,rand=T)","36f6":"FT(wt=0)","b5e6":"FT(wt=3,rand=F)'","pret":"Orig.","1146":"FT(wt=3,rand=T)'"}
    # lnames = {"10dc":"FT(wt=3,rand=T,SWA=F)","0b31":"FT(wt=0)",
    #           "05bf":"FT(wt=3,rand=T,SWA=T)","pret":"Orig.","aa54":"tgt_FT(wt=3,rand=T)",
    #           "a2ba":"(SWA=T,lr=1e-3)@21","18e9":"Ours",
    #           "2862":"NoScaleNoRand","f628":"NoScaleRand",
    #           "2841":"CropModeNo","b789":"CropModeYes",
    #           "a9b5":"svnlb","9df0":"NoFlow","1d4d":"lr1e-4",
    #           "5544":"NoFlip"}
    # lnames = {"f43e":"FT(wt=1,rand=T)","0b31":"FT(wt=0)",
    #           "7df7":"FT(wt=1,rand=F)","pret":"Orig.","aa54":"FT(wt=3,rand=T)"}
    lnames = {"55d5=29":"FT(wt=0)","bef4=29":"FT(wt=3,No Flow)",
              "3456=29":"FT(wt=3,With Flow,No Rand)","2027-04":"FT(wt=3,Flow,Rand)"}
    records['prefix'] = records['pretrained_path'].str.slice(0,4)
    records['postfix'] = records['pretrained_path'].str.slice(-8,-5)
    records['pretrained_path'] = records['prefix'] + records['postfix']
    records['strred'] = records['strred']*100.
    ppath = records['pretrained_path'].to_numpy()
    labels = np.array(ppath)
    for label,name in lnames.items():
        labels = np.where(ppath==label,name,labels)
    records['labels'] = labels
    gfields = ['labels','sigma','wt']
    fmts = ["%2.2f","%.3f","%2.2f"]
    afields = ['psnrs','ssims','strred']
    agg = lambda x: np.mean(np.stack(x))
    records = records.groupby(gfields).agg({k:agg for k in afields})
    records = records.reset_index()
    records = records.sort_values(by="psnrs")
    for _,row in records.iterrows():
        g = [row[g] for g in gfields]
        i_fxn = lambda f,fmt: fmt % (row[f])
        print(list(g),"/".join([i_fxn(f,fmt) for f,fmt in zip(afields,fmts)]))

    # for g,gdf in records.groupby(gfields):
    #     i_fxn = lambda f,fmt: fmt % (gdf[f].to_numpy()[0])
    #     print(list(g),"/".join([i_fxn(f,fmt) for f,fmt in zip(afields,fmts)]))

if __name__ == "__main__":
    main()
