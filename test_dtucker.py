from inc.tucker import *
import itertools

def mat_mat_mul(A, B):
    return (A[:, None, :] * B.transpose(0, 1)[None, :, :]).sum(dim = 2)

class DTucker:
    @torch.no_grad()
    def __init__(self, X, ranks, tol, maxiters):
        self.tol = tol
        self.maxiters = maxiters
        self.tlen = X.size(dim = 0)
        self.n_dims = len(ranks)
        assert self.n_dims >= 3
        size_max = max(X.size()) + 1
        self.perm = sorted(range(self.n_dims), key = lambda dim: -(size_max if dim == 0 else X.size(dim)))
        self.perm_X = X.permute(*self.perm)
        self.perm_sizes = tuple(self.perm_X.size())
        self.perm_ranks = np.array(ranks)[self.perm].tolist()
        self.norm2 = tensor_norm2(self.perm_X).item()
        self.inv_perm = [None for p in range(self.n_dims)]
        for i, p in enumerate(self.perm):
            self.inv_perm[p] = i
        self.unsqueezer = tuple(None if p >= 2 else slice(None) for p in range(self.n_dims))
        
        # D-decomp
        self.decomp_rank = sorted(ranks)[1]
        U, S, VH = tlin.svd(self.perm_X.permute(*(list(range(2, self.n_dims)) + [0, 1])), full_matrices = (self.decomp_rank > min(self.perm_sizes[: 2])))
        self.U = U.permute(*([self.n_dims - 2, self.n_dims - 1] + list(range(self.n_dims - 2))))[:, : self.decomp_rank].clone().detach()
        self.S = S.permute(*([self.n_dims - 2] + list(range(self.n_dims - 2))))[: self.decomp_rank].clone().detach()
        self.V = VH.permute(*([self.n_dims - 1, self.n_dims - 2] + list(range(self.n_dims - 2))))[:, : self.decomp_rank].clone().detach()
    def iter_slices(self):
        return itertools.product(*[range(siz) for siz in self.perm_sizes[2 :]])
    def init_tucker(self, U, perm_ranks): # D-init
        A = [None for p in range(self.n_dims)]
        US = U * self.S
        _, A[0] = mat_svd(US.flatten(start_dim = 1), perm_ranks[0])
        VS = mat_mat_mul(self.V, mat_mat_mul(US.transpose(0, 1), A[0][self.unsqueezer]))
        _, A[1] = mat_svd(VS.flatten(start_dim = 1), perm_ranks[1])
        Y = mat_mat_mul(VS.transpose(0, 1), A[1][self.unsqueezer])
        for p in range(2, self.n_dims):
            Y_mat = tensor_unfold(Y, dim = p)
            _, A[p] = mat_svd(Y_mat, perm_ranks[p])
            Y = tensor_mat_mul(Y, A[p].T, dim = p)
        return A
    def update_tucker(self, U, A): # D-update
        AT = [Ap.T for Ap in A]
        B = [None for p in range(self.n_dims)]
        # first mode
        Y0 = mat_mat_mul(U * self.S, mat_mat_mul(AT[1][self.unsqueezer], self.V).transpose(0, 1))
        Y0 = tensor_mats_mul(Y0, A_dim_list = [(AT[p], p) for p in range(2, self.n_dims)])
        _, B[0] = mat_svd(tensor_unfold(Y0, dim = 0), AT[0].size(dim = 0))
        # second mode
        Y1 = mat_mat_mul((B[0].T)[self.unsqueezer], U)
        Y1_inter = Y1
        Y1 = tensor_mats_mul(mat_mat_mul(Y1 * self.S, self.V.transpose(0, 1)), A_dim_list = [(AT[p], p) for p in range(2, self.n_dims)])
        _, B[1] = mat_svd(tensor_unfold(Y1, dim = 1), AT[1].size(dim = 0))
        # other modes
        Yp = mat_mat_mul(Y1_inter * self.S, mat_mat_mul((B[1].T)[self.unsqueezer], self.V).transpose(0, 1))
        Yp_inter = Yp
        for p in range(2, self.n_dims):
            Yp = tensor_mats_mul(Yp, A_dim_list = [(B[q].T if q < p else AT[q]) for q in range(2, self.n_dims) if q != p])
            _, B[p] = mat_svd(tensor_unfold(Yp, dim = p), AT[p].size(dim = 0))
        G = tensor_mats_mul(Yp_inter, A_dim_list = [(B[p].T, p) for p in range(2, self.n_dims)])
        return G, B
    @torch.no_grad()
    def query_tucker(self, t0, t1): # [t0, t1)
        norm2 = self.norm2 - tensor_norm2(self.U[: t0] * self.S).item() - tensor_norm2(self.U[t1 :] * self.S).item()
        perm_ranks = deepcopy(self.perm_ranks)
        perm_ranks[0] = min(perm_ranks[0], t1 - t0)
        U = self.U[t0 : t1]
        A = self.init_tucker(U, perm_ranks)
        fit = 0.
        for it in range(self.maxiters):
            G, A = self.update_tucker(U, A)
            fit_old = fit
            fit = tensor_norm2(G) / norm2
            if it > 0 and abs(fit - fit_old) < args.tol:
                break
        return Tucker(G = G.permute(*self.inv_perm), U = [A[i] for i in self.inv_perm]), fit

from inc.args import *

parser = ArgParser(prog = DTucker.__name__)
args = parser.parse_args()

from inc.data import *

X = load_data(root = args.data_root, name = args.dataset, device = args.device)
tlen = X.size(dim = 0)

mem0 = torch.cuda.memory_allocated(0)
tic = time.time()
prep = DTucker(X = X, ranks = args.ranks, tol = args.tol, maxiters = args.maxiters)
toc = time.time()
mem1 = torch.cuda.memory_allocated(0)
res = dict(dur = toc - tic, mem = (mem1 - mem0) / 1073741824)
with open(f'{args.save_name}~eval~prep.pkl', 'wb') as fo:
    pkl.dump(res, fo)

queries = load_queries(root = args.queries_root, name = args.dataset)
res = Dict(qlen = [], orig = [], err = [], dur = [])
with torch.no_grad():
    for qlen, ts in queries.items():
        for t0, t1 in tqdm(ts, desc = f'qlen={qlen}'):
            tic = time.time()
            tucker, _ = prep.query_tucker(t0, t1)
            toc = time.time()
            res.qlen.append(qlen)
            Xq = X[t0 : t1]
            res.orig.append(tensor_norm(Xq).item())
            res.err.append(tensor_norm(Xq - tensor_mats_mul(tucker.G, A_dim_list = [(Up, p) for p, Up in enumerate(tucker.U)])).item())
            res.dur.append(toc - tic)
        print(f'[qlen={qlen}] avg_err={(np.array(res.err[-len(ts) :]) / np.array(res.orig[-len(ts) :])).mean():.4f}', flush = True)
res = dict(res)
with open(f'{args.save_name}~eval~query.pkl', 'wb') as fo:
    pkl.dump(res, fo)
