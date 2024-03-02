from inc.tensor import *

Tucker = namedtuple('Tucker', 'G U') # core G, factors U

@torch.no_grad()
def tucker_zeros(sizes, ranks, dtype, device):
    """Tucker decomposition of an empty tensor"""
    return Tucker(G = torch.zeros(*ranks, dtype = dtype, device = device), U = [torch.zeros(sizes[p], ranks[p], dtype = dtype, device = device) for p in range(len(sizes))]), 1.

@torch.no_grad()
def tucker_als(X, ranks, tol, maxiters, norm2 = None, verbose = False, mat_svd_fn = mat_svd): # pads zeros if rank > size
    """
    Tucker-ALS for Tucker decomposition
    adapted from the MATLAB implementation at https://gitlab.com/tensors/tensor_toolbox/-/blob/dev/tucker_als.m
    """
    n_dims = len(ranks)
    assert n_dims > 1
    if norm2 is None:
        norm2 = tensor_norm2(X)
    if norm2 <= 0.:
        return tucker_zeros(sizes = tuple(X.size()), ranks = ranks, dtype = X.dtype, device = X.device)
    sizes = list(X.size())
    n_vecs = np.minimum(ranks, sizes).tolist()
    U = [None for p in range(n_dims)]
    dims = sorted(range(n_dims), key = lambda p: -sizes[p])
    for p in dims[1 :]:
        U[p] = torch.rand(sizes[p], n_vecs[p], dtype = X.dtype, device = X.device)
    pbar = range(maxiters)
    if verbose:
        pbar = tqdm(pbar)
    fit = 0.
    for it in pbar:
        for p in dims:
            Y = tensor_mats_mul(X, A_dim_list = [(U[q].T, q) for q in range(n_dims) if q != p])
            U[p] = mat_svd_fn(tensor_unfold(Y, dim = p), top = n_vecs[p])[1]
        G = tensor_mat_mul(Y, U[p].T, dim = p)
        fit_old = fit
        fit = tensor_norm2(G) / norm2
        if verbose:
            pbar.set_description(f'it={it}, fit={fit:.4f}')
        if it > 0 and abs(fit - fit_old) < tol:
            break
    if np.less(n_vecs, ranks).any():
        G = tensor_pad(G, ranks)
        for p in range(n_dims):
            if n_vecs[p] < ranks[p]:
                U[p] = tensor_pad(U[p], (sizes[p], ranks[p]))
    return Tucker(G = G, U = U), fit

@torch.no_grad()
def tucker_stitch(tuckers, ranks, tol, maxiters, norm2, verbose = False, mat_svd_fn = mat_svd):
    """stitch subtensor Tucker decompositions along the first mode"""
    n_dims = len(ranks)
    assert n_dims > 1
    sizes = [Up.size(dim = 0) for Up in tuckers[0].U]
    for i in range(1, len(tuckers)):
        sizes[0] += tuckers[i].U[0].size(dim = 0)
    n_vecs = np.minimum(ranks, sizes).tolist()
    dtype = tuckers[0].G.dtype
    device = tuckers[0].G.device
    if norm2 <= 0.:
        return tucker_zeros(sizes = sizes, ranks = ranks, dtype = dtype, device = device)
    U = [None for p in range(n_dims)]
    dims = sorted(range(n_dims), key = lambda p: -sizes[p])
    for p in dims[1 :]:
        U[p] = torch.rand(sizes[p], n_vecs[p], dtype = dtype, device = device)
    pbar = range(maxiters)
    if verbose:
        pbar = tqdm(pbar)
    fit = 0.
    for it in pbar:
        # update non-temporal factor matrices
        for p in dims:
            if p == 0: # the temporal mode
                Y = torch.cat([
                    tensor_unfold(tensor_mats_mul(tucker.G, A_dim_list = [
                        ((U[q].T.mm(tucker.U[q]) if q != p else tucker.U[q]), q) for q in range(n_dims)
                    ]), dim = p) for tucker in tuckers
                ], dim = 0)
            else: # non-temporal modes
                Y = 0.
                t0 = 0
                for tucker in tuckers:
                    t1 = t0 + tucker.U[0].size(dim = 0)
                    Y = Y + tensor_unfold(tensor_mats_mul(tucker.G, A_dim_list = [
                        (((U[q] if q != 0 else U[q][t0 : t1]).T.mm(tucker.U[q]) if q != p else tucker.U[q]), q) for q in range(n_dims)
                    ]), dim = p)
                    t0 = t1
            U[p] = mat_svd_fn(Y, top = n_vecs[p])[1]
        # reusing Y to compute the core tensor
        G = tensor_fold(U[p].T.mm(Y), dim = p, sizes = n_vecs)
        fit_old = fit
        fit = tensor_norm2(G) / norm2
        if verbose:
            pbar.set_description(f'it={it}, fit={fit:.4f}')
        if it > 0 and abs(fit - fit_old) < tol:
            break
    if np.less(n_vecs, ranks).any():
        G = tensor_pad(G, ranks)
        for p in range(n_dims):
            if n_vecs[p] < ranks[p]:
                U[p] = tensor_pad(U[p], (sizes[p], ranks[p]))
    return Tucker(G = G, U = U), fit

@torch.no_grad()
def tucker_partial(tucker, t0, t1, qr = True): # [t0, t1)
    """approximate subtensor Tucker decomposition"""
    G = tucker.G
    U = [Up.clone() if p != 0 else Up for p, Up in enumerate(tucker.U)]
    if qr:
        Q, R = tlin.qr(U[0][t0 : t1], mode = 'reduced')
        rank = U[0].size(dim = 1)
        if t1 - t0 < rank:
            Q = tensor_pad(Q, sizes = (t1 - t0, rank))
            R = tensor_pad(R, sizes = (rank, rank))
        U[0] = Q
        G = tensor_mat_mul(G, A = R, dim = 0)
    else:
        G = G.clone()
        U[0] = U[0][t0 : t1].clone()
    return Tucker(G = G, U = U)
