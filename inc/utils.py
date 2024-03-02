from inc.header import *

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed ^ 1)
    torch.manual_seed(seed ^ 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed ^ 3)
