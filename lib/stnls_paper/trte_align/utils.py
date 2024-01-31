

import sys
import copy
import datetime
from easydict import EasyDict as edict

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
    dname_template = {'dname': {'psnrs': [],
                                'ssims': [],
                                'best_psnr': {
                                    'value': 0.0,
                                    'epoch': 0
                                },
                                'best_ssim': {
                                    'value': 0.0,
                                    'epoch': 0
                                }}}
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

def calc_psnr(vid,flows):
    pass

def calc_ssim(vid,flows):
    pass
