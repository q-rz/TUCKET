from inc.tucker import *

class TucketNode:
    def __init__(self, t0, t1, l = None, r = None, built = None, norm2 = 0., tucker = None):
        self.t0 = t0
        self.t1 = t1
        self.l = l
        self.r = r
        self.built = built
        self.norm2 = norm2
        self.tucker = tucker
    @property
    def tm(self):
        return (self.t0 + self.t1) >> 1
    @property
    def tlen(self):
        return self.t1 - self.t0
    def build(self, X, ranks, tol, maxiters):
        self.norm2 = tensor_norm2(X)
        self.tucker, fit = tucker_als(X, ranks, tol, maxiters, norm2 = self.norm2, verbose = False) # not using mat_svd_eigh due to its numerical instability
        self.built = True
    def stitch(self, ranks, tol, maxiters):
        self.norm2 = self.l.norm2 + self.r.norm2
        self.tucker, fit = tucker_stitch([self.l.tucker, self.r.tucker], ranks, tol, maxiters, norm2 = self.norm2, verbose = False)
        self.built = True
    def partial_tucker(self, t0, t1, qr = True):
        return tucker_partial(self.tucker, t0 = max(t0, self.t0) - self.t0, t1 = min(t1, self.t1) - self.t0, qr = qr)

class TucketTree: # time starts from 0
    def __init__(self, ranks, tol, maxiters, alloc = 1):
        self.cfg = Dict(ranks = deepcopy(ranks), tol = tol, maxiters = maxiters)
        self.n_dims = len(ranks)
        self.tlen = 0
        self.root = None
        self.n_nodes = 0
        self.alloc = alloc
        self.nodes = [None for _ in range(2 ** (math.ceil(math.log2(alloc)) + 1) - 1)]
        ##self.hits_logs = []
    def _new_node(self, t0, t1, **kwargs):
        node_id = self.n_nodes
        self.n_nodes += 1
        if node_id >= len(self.nodes):
            self.nodes.append(None)
        self.nodes[node_id] = TucketNode(t0 = t0, t1 = t1, **kwargs)
        return self.nodes[node_id]
    def _insert(self, node, t, Xt):
        if t == node.t0 and t + 1 == node.t1: # leaf node
            node.build(X = Xt, **self.cfg)
        elif t < node.tm: # go to left child
            if node.l is None:
                node.l = self._new_node(node.t0, node.tm)
            self._insert(node.l, t, Xt)
        else: # go to right child
            if node.r is None:
                node.r = self._new_node(node.tm, node.t1)
            self._insert(node.r, t, Xt)
            if t + 1 == node.t1:
                node.stitch(**self.cfg)
    def append(self, Xt):
        t = self.tlen
        self.tlen += 1
        if self.root is None:
            self.root = self._new_node(t0 = 0, t1 = 1)
        elif t >= self.root.t1:
            old_root = self.root
            self.root = self._new_node(t0 = 0, t1 = old_root.t1 << 1, l = old_root)
        self._insert(self.root, t, Xt[None])
    def _recall(self, node, t0, t1, prune, hits: list):
        if node.built and (t1 - t0) >= node.tlen * prune:
            hits.append(node)
        elif t1 <= node.tm:
            self._recall(node.l, t0, t1, prune, hits)
        elif t0 >= node.tm:
            self._recall(node.r, t0, t1, prune, hits)
        else:
            self._recall(node.l, t0, node.tm, prune, hits)
            self._recall(node.r, node.tm, t1, prune, hits)
    def _query_norm2(self, node, t0, t1):
        if t0 == node.t0 and t1 == node.t1:
            return node.norm2
        elif t1 <= node.tm:
            return self._query_norm2(node.l, t0, t1)
        elif t0 >= node.tm:
            return self._query_norm2(node.r, t0, t1)
        else:
            return self._query_norm2(node.l, t0, node.tm) + self._query_norm2(node.r, node.tm, t1)
    def query_tucker(self, t0, t1, prune): # [t0, t1)
        norm2 = self._query_norm2(self.root, t0, t1)
        hits = []
        self._recall(self.root, t0 = t0, t1 = t1, prune = prune, hits = hits)
        tucker, fit = tucker_stitch(tuckers = [
            node.tucker if t0 <= node.t0 and t1 >= node.t1 else node.partial_tucker(t0, t1, qr = len(hits) < 2)
            for node in hits
        ], norm2 = norm2, **self.cfg, mat_svd_fn = mat_svd_eigh)
        return tucker, fit, len(hits)

from inc.args import *

parser = ArgParser(prog = TucketTree.__name__)
parser.add_argument('--prune', type = float, default = 0.7, help = 'pruning threshold of TUCKET')
args = parser.parse_args()

from inc.data import *

X = load_data(root = args.data_root, name = args.dataset, device = args.device)
tlen = X.size(dim = 0)

tree = TucketTree(ranks = args.ranks, tol = args.tol, maxiters = args.maxiters, alloc = tlen)
res = Dict(dur = [], mem = [])
tbar = trange(tlen)
for t in tbar:
    mem0 = torch.cuda.memory_allocated(0)
    tic = time.time()
    tree.append(X[t])
    toc = time.time()
    mem1 = torch.cuda.memory_allocated(0)
    res.dur.append(toc - tic)
    res.mem.append((mem1 - mem0) / 1073741824)
    tbar.set_description(f'cummem={mem1 / 1073741824:.2f}GB')
res = dict(res)
with open(f'{args.save_name}~eval~append.pkl', 'wb') as fo:
    pkl.dump(res, fo)

queries = load_queries(root = args.queries_root, name = args.dataset)
res = Dict(qlen = [], orig = [], err = [], dur = [], hits = [])
with torch.no_grad():
    for qlen, ts in queries.items():
        for t0, t1 in tqdm(ts, desc = f'qlen={qlen}'):
            tic = time.time()
            tucker, _, hits = tree.query_tucker(t0, t1, prune = args.prune)
            toc = time.time()
            res.qlen.append(qlen)
            Xq = X[t0 : t1]
            res.orig.append(tensor_norm(Xq).item())
            res.err.append(tensor_norm(Xq - tensor_mats_mul(tucker.G, A_dim_list = [(Up, p) for p, Up in enumerate(tucker.U)])).item())
            res.dur.append(toc - tic)
            res.hits.append(hits)
        print(f'[qlen={qlen}] avg_err={(np.array(res.err[-len(ts) :]) / np.array(res.orig[-len(ts) :])).mean():.4f}', flush = True)
res = dict(res)
res = dict(res)
with open(f'{args.save_name}~eval~query.pkl', 'wb') as fo:
    pkl.dump(res, fo)
