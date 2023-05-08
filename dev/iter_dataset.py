import tqdm
import torch as th
import data_hub
from easydict import EasyDict as edict

def main():

    R = 3 # number of times to check
    cfg = edict()
    cfg.dname = "iphone_s2023"
    cfg.iphone_type = "structures"
    # cfg.iphone_type = "texture"
    cfg.video_seq_max = 120
    # cfg.dname = "davis_cropped"
    cfg.nsamples_tr = 0#100
    cfg.nframes = 0
    cfg.sigma = 30
    data,loaders = data_hub.sets.load(cfg)
    keys = data.tr.paths['images'].keys()
    print(sum([len(data.tr.paths['images'][k]) for k in keys]))
    # print([len(data['tr'][i]['noisy']) for i in range(len(data['tr']))])
    # loader = loaders['tr']
    # L = len(loader)
    # print("L: ",L)
    return

    # -- gather indices --
    indices = th.zeros((R,L)).int()
    for r in range(R):
        for l,sample in enumerate(loader):
            indices[r,l] = sample['index']

    # -- summarize --
    indices = th.sort(indices,1).values
    # print(indices)
    # print(th.any(th.abs(indices[0] - indices[1]) != 0))
    neq = 1.*(th.abs(indices[0] - indices[1]) != 0)
    print("Percent not equal: ",th.mean(neq).item())


if __name__ == "__main__":
    main()
