from inc.tucker import *

class ZoomBlock:
    """block of Zoom-Tucker"""
    @torch.no_grad()
    def __init__(self, t0, t1, X, ranks, tol, maxiters):
        self.t0 = t0
        self.t1 = t1
        self.norm2 = tensor_norm2(X)
        self.tucker, fit = tucker_als(X, ranks, tol, maxiters, norm2 = self.norm2, verbose = False)
    def partial_tucker(self, t0, t1, qr = True):
        return tucker_partial(self.tucker, t0 = max(t0, self.t0) - self.t0, t1 = min(t1, self.t1) - self.t0, qr = qr)

class ZoomTucker:
    """
    Zoom-Tucker
    adapted from the authors' MATLAB implementation at https://datalab.snu.ac.kr/zoomtucker/
    """
    @torch.no_grad()
    def __init__(self, X, block_size, ranks, tol, maxiters):
        """preprocess blocks"""
        self.cfg = Dict(ranks = deepcopy(ranks), tol = tol, maxiters = maxiters)
        self.block_size = block_size
        self.n_dims = len(ranks)
        self.X = X
        self.T = X.size(dim = 0)
        self.blocks = [ZoomBlock(t0, min(t0 + self.block_size, self.T), X = X[t0 : min(t0 + self.block_size, self.T)], **self.cfg) for t0 in trange(0, self.T, self.block_size)]
    def query_tucker(self, t0, t1): # [t0, t1)
        """answer a range query by stitching blocks"""
        hits = [self.blocks[i] for i in range(t0 // self.block_size, (t1 - 1) // self.block_size + 1)]
        tuckers = [
            block.tucker if t0 <= block.t0 and t1 >= block.t1 else block.partial_tucker(t0, t1, qr = True)
            for block in hits
        ]
        norm2 = 0.
        for block, tucker in zip(hits, tuckers):
            if t0 <= block.t0 and t1 >= block.t1:
                norm2 += block.norm2
            else:
                norm2 += tensor_norm2(tucker.G)
        tucker, fit = tucker_stitch(tuckers = tuckers, norm2 = norm2, **self.cfg)
        return tucker, fit, len(hits)

from inc.args import *

parser = ArgParser(prog = ZoomTucker.__name__)
parser.add_argument('--block', type = int, help = 'block size of Zoom-Tucker')
args = parser.parse_args()

from inc.data import *

X = load_data(root = args.data_root, name = args.dataset, device = args.device)
tlen = X.size(dim = 0)

mem0 = torch.cuda.memory_allocated(0)
tic = time.time()
prep = ZoomTucker(X = X, block_size = args.block, ranks = args.ranks, tol = args.tol, maxiters = args.maxiters)
toc = time.time()
mem1 = torch.cuda.memory_allocated(0)
res = dict(dur = toc - tic, mem = (mem1 - mem0) / 1073741824)
with open(f'{args.save_name}~eval~prep.pkl', 'wb') as fo:
    pkl.dump(res, fo)

queries = load_queries(root = args.queries_root, name = args.dataset)
res = Dict(qlen = [], orig = [], err = [], dur = [], hits = [])
with torch.no_grad():
    for qlen, ts in queries.items():
        for t0, t1 in tqdm(ts, desc = f'qlen={qlen}'):
            tic = time.time()
            tucker, _, hits = prep.query_tucker(t0, t1)
            toc = time.time()
            res.qlen.append(qlen)
            Xq = X[t0 : t1]
            res.orig.append(tensor_norm(Xq).item())
            res.err.append(tensor_norm(Xq - tensor_mats_mul(tucker.G, A_dim_list = [(Up, p) for p, Up in enumerate(tucker.U)])).item())
            res.dur.append(toc - tic)
            res.hits.append(hits)
        print(f'[qlen={qlen}] avg_err={(np.array(res.err[-len(ts) :]) / np.array(res.orig[-len(ts) :])).mean():.4f}', flush = True)
res = dict(res)
with open(f'{args.save_name}~eval~query.pkl', 'wb') as fo:
    pkl.dump(res, fo)
