from inc.args import *

parser = ArgParser(prog = 'TuckerALS')
args = parser.parse_args()

from inc.data import *
from inc.tucker import *

X = load_data(root = args.data_root, name = args.dataset, device = args.device)
tlen = X.size(dim = 0)

queries = load_queries(root = args.queries_root, name = args.dataset)
res = Dict(qlen = [], orig = [], err = [], dur = [])
with torch.no_grad():
    for qlen, ts in queries.items():
        for t0, t1 in tqdm(ts, desc = f'qlen={qlen}'):
            Xq = X[t0 : t1]
            tic = time.time()
            tucker, _ = tucker_als(Xq, args.ranks, args.tol, args.maxiters, verbose = False)
            toc = time.time()
            res.qlen.append(qlen)
            res.orig.append(tensor_norm(Xq).item())
            res.err.append(tensor_norm(Xq - tensor_mats_mul(tucker.G, A_dim_list = [(Up, p) for p, Up in enumerate(tucker.U)])).item())
            res.dur.append(toc - tic)
        print(f'[qlen={qlen}] avg_err={(np.array(res.err[-len(ts) :]) / np.array(res.orig[-len(ts) :])).mean():.4f}', flush = True)
res = dict(res)
with open(f'{args.save_name}~eval~query.pkl', 'wb') as fo:
    pkl.dump(res, fo)
