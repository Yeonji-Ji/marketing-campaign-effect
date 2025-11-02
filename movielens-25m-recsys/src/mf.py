import scipy.sparse as sp

n_users = train["u"].max() + 1
n_items = train["i"].max() + 1

R = sp.coo_matrix(
    (train["rating"], (train["u"], train["i"])),
    shape=(n_users, n_items)
).tocsr()

K = 64
u_f, s, vt = svds(R.astype("float32"), k=K)

S = np.diag(s)
U = u_f @ S 
V = vt.T         

def mf_score(u_idx, i_idx):
    return (U[u_idx] * V[i_idx]).sum(axis=1)


def score_get_topk(df, k=100):
    out = []
    for u, g in df.groupby(["u"]):
        g = g.copy()
        g["score"] = mf_score(g["u"].values, g["i"].values)
        out.append(g.sort_values("score", ascending=False).head(k))

    return pd.concat(out, axis=0).reset_index(drop=True)