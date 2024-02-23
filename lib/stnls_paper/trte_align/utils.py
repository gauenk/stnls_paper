

import sys
import copy
import datetime
from easydict import EasyDict as edict
from einops import rearrange

from dev_basics.utils.metrics import compute_psnrs,compute_ssims

def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content

def get_stat_dict(dnames):
    dname_template = {'psnrs': [],
                      'ssims': [],
                      'best_psnr': {
                          'value': 0.0,
                          'epoch': 0
                      },
                      'best_ssim': {
                          'value': 0.0,
                          'epoch': 0
                      }}
    # dname_template = {'dname': {'psnrs': [],
    #                             'ssims': [],
    #                             'best_psnr': {
    #                                 'value': 0.0,
    #                                 'epoch': 0
    #                             },
    #                             'best_ssim': {
    #                                 'value': 0.0,
    #                                 'epoch': 0
    #                             }}}
    stat_dict = {'epochs': 0,'losses': [],'ema_loss': 0.0}
    for dname in dnames:
        stat_dict[dname] = copy.deepcopy(dname_template)
    return stat_dict

class ExperimentLogger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')
    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def calc_psnr(vid,stack):
    vid = vid[:,None].repeat(1,2,1,1,1,1)
    vid = rearrange(vid,'b k t c h w -> b (k t) c h w')
    stack = rearrange(stack,'b 1 k t c h w -> b (k t) c h w')
    psnrs = compute_psnrs(stack,vid)
    return psnrs

def calc_ssim(vid,stack):
    vid = vid[:,None].repeat(1,2,1,1,1,1)
    vid = rearrange(vid,'b k t c h w -> b (k t) c h w')
    stack = rearrange(stack,'b 1 k t c h w -> b (k t) c h w')
    ssims = compute_ssims(stack,vid)
    return ssims
