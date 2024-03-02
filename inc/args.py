from inc.utils import *
from inc.data import *

def parse_list(list_str, cls):
    return list(map(cls, list_str.split(',')))

def parse_ints(ints_str):
    return parse_list(ints_str, int)

class ArgParser(argparse.ArgumentParser):
    def __init__(self, prog, **kwargs):
        super().__init__(prog = prog, **kwargs)
        self._prog = prog
        self.add_argument('--device', type = torch.device, help = 'device for PyTorch (e.g., cuda:0)')
        self.add_argument('--dataset', type = str, choices = list(DATA_FNS.keys()), help = 'name of the dataset')
        self.add_argument('--ranks', type = parse_ints, help = 'target sizes of Tucker decomposition (comma-separated; no spaces)')
        self.add_argument('--tol', type = float, default = 1e-2, help = 'tolerence in the stopping criterion for Tucker decomposition')
        self.add_argument('--maxiters', type = int, default = 20, help = 'maximum number of ALS iterations for Tucker decomposition')
        self.add_argument('--data_root', type = str, default = 'inputs', help = 'path to the data folder')
        self.add_argument('--queries_root', type = str, default = 'inputs', help = 'path to the queries folder')
        self.add_argument('--seed', type = int, default = 998244353, help = 'random seed')
        self.add_argument('--save_name', type = str, default = osp.join('outputs', self._prog), help = 'filename prefix for saving the outputs (including folders)')
    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        set_seed(args.seed)
        return Dict(vars(args))
