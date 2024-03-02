from inc.header import *

DATA_FNS = dict(
    AirQuality = lambda root: np.load(osp.join(root, 'AirQuality.npy')),
    Traffic = lambda root: np.load(osp.join(root, 'Traffic.npy')),
    USStock = lambda root: np.load(osp.join(root, 'USStock.npy')),
    KRStock = lambda root: np.load(osp.join(root, 'KRStock.npy')),
)

def load_data(root, name, device):
    return torch.tensor(DATA_FNS[name](root).astype(np.float32), device = device)

def load_queries(root, name):
    with open(osp.join(root, f'{name}~queries.json'), 'r') as fi:
        queries = json.load(fi)
    return {int(qlen): [(int(t0), int(t1)) for t0, t1 in ts] for qlen, ts in queries.items()}
