"""A1 大道至简极限版: SAGE(3000) + 特征工程(1000)
5-6小时, 无GCN/GAT/Node2Vec噪声
核心: 特征工程与SAGE分歧56.6% → 加权投票挖掘互补信号
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, os, sys, time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_norm, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter
import threading, itertools

DEV = torch.device('cuda')
SEED_BASE = 42
np.random.seed(SEED_BASE)
torch.manual_seed(SEED_BASE)

# ═══ DATA ═══
DATA = sys.argv[1] if len(sys.argv) > 1 else 'data/cls_data/A1.npz'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'a1_sage_feat_out'
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, '.ckpt.npz')

print(f'[LOAD] {DATA}', flush=True)
d = np.load(DATA)
adj_raw = csr_matrix((d['adj_data'], d['adj_indices'], d['adj_indptr']), shape=tuple(d['adj_shape']))
feat_raw = csr_matrix((d['attr_data'], d['attr_indices'], d['attr_indptr']), shape=tuple(d['attr_shape']))
labels = d['labels'].astype(int)
tr_idx = d['train_idx']
te_idx = d['test_idx']
N, NC = adj_raw.shape[0], 10
N_TEST = len(te_idx)
print(f'  N={N} test={N_TEST}', flush=True)

# Features
feat = sk_norm(feat_raw.toarray().astype(np.float32), norm='l2', axis=1)
feat[np.isnan(feat).any(axis=1)] = 0
F_BASE = feat.shape[1]
deg = np.array(adj_raw.sum(1)).flatten().astype(np.float32)
deg_feat = np.hstack([np.log1p(deg).reshape(-1,1),
    (deg<5).astype(np.float32).reshape(-1,1), (deg<10).astype(np.float32).reshape(-1,1)])

# Normalized adjacency
asym = adj_raw + adj_raw.T; asym.setdiag(1)
ds = np.array(asym.sum(1)).flatten()
dis = np.where(ds>0, 1.0/np.sqrt(ds), 0)
An = csr_matrix((dis,(range(N),range(N))),shape=(N,N)) @ asym @ csr_matrix((dis,(range(N),range(N))),shape=(N,N))
coo = An.tocoo()
A_sp = torch.sparse_coo_tensor(
    torch.tensor(np.vstack((coo.row,coo.col)), dtype=torch.long, device=DEV),
    torch.tensor(coo.data, dtype=torch.float32, device=DEV), (N,N)).coalesce()

# LP transition
rs = np.array(asym.sum(1)).flatten(); rs[rs==0]=1
T_p = csr_matrix((1.0/rs,(range(N),range(N))),shape=(N,N)) @ asym

Y_t = torch.tensor(labels, dtype=torch.long, device=DEV)
tr_t = torch.tensor(tr_idx, dtype=torch.long, device=DEV)
te_t = torch.tensor(te_idx, dtype=torch.long, device=DEV)
Y0 = np.zeros((N,NC), dtype=np.float32)
for ti in tr_idx: Y0[ti, labels[ti]] = 1.0

# LP: 9 variants
ALPHAS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
LP = {}
for a in ALPHAS:
    Y = Y0.copy()
    for _ in range(80): Y = a*(T_p@Y)+(1-a)*Y0
    LP[a] = Y.astype(np.float32)
lp_probs = np.stack([LP[a][te_idx] for a in ALPHAS], axis=0)
print(f'  LP: {lp_probs.shape}', flush=True)
F_IN = F_BASE + NC + 3

# ═══ SAGE TRAINER ═══
EPOCHS = 400; LR = 0.005; WD = 5e-4; DROP = 0.5
lock = threading.Lock()
X_cache = {}

def get_X(a):
    if a not in X_cache:
        X_cache[a] = torch.tensor(np.hstack([feat, LP[a], deg_feat]).astype(np.float32), device=DEV)
    return X_cache[a]

def train_sage(hdim, nlayers, alpha, seed, X):
    torch.manual_seed(seed)
    layers = [nn.Linear((F_IN*2) if i==0 else (hdim*2), hdim if i<nlayers-1 else NC, device=DEV) for i in range(nlayers)]
    opt = torch.optim.AdamW([p for l in layers for p in l.parameters()], lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        for l in layers: l.train()
        opt.zero_grad()
        h = X
        for i in range(nlayers-1):
            h = F.relu(F.dropout(layers[i](torch.cat([h, A_sp@h], -1)), p=DROP, training=True))
        F.cross_entropy(layers[-1](torch.cat([h, A_sp@h], -1))[tr_t], Y_t[tr_t]).backward(); opt.step()
    for l in layers: l.eval()
    with torch.no_grad():
        h = X
        for i in range(nlayers-1):
            h = F.relu(F.dropout(layers[i](torch.cat([h, A_sp@h], -1)), p=DROP, training=False))
        return F.softmax(layers[-1](torch.cat([h, A_sp@h], -1)), dim=-1).cpu().numpy()[te_idx]

# ═══ CKPT ═══
all_ckpt = {}
if os.path.exists(CKPT):
    ckpt = np.load(CKPT, allow_pickle=True)
    all_ckpt = {k: ckpt[k] for k in ckpt.files}
    print(f'[CKPT] {len(all_ckpt)} saved', flush=True)

def save_ckpt():
    np.savez_compressed(CKPT, **all_ckpt)

# ═══════════════════════════════════════════════
# SAGE: 1000 configs × 3 seeds = 3000 voters (~5h)
# ═══════════════════════════════════════════════
print('\n' + '='*60)
print('SAGE: 3000 voters (~5 hours)')
print('='*60, flush=True)

# Grid: 25 hdims × 5 layers × 9 alphas = 1125 configs → subsample to 1000
sage_hdims = [16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024]
sage_layers = [1, 2, 3, 4, 5]
sage_alphas = [0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
sage_seeds = 3

sage_jobs = []
for hd in sage_hdims:
    for nl in sage_layers:
        for al in sage_alphas:
            if nl == 1 and hd > 512: continue
            if nl >= 4 and hd < 48: continue
            if nl >= 5 and hd < 96: continue
            for s in range(sage_seeds):
                key = f's_h{hd}_L{nl}_a{al:.2f}_s{s}'
                if key not in all_ckpt:
                    sage_jobs.append((hd, nl, al, s, key))

if len(sage_jobs) > 1000:
    np.random.shuffle(sage_jobs)
    sage_jobs = sage_jobs[:1000]
print(f'  Jobs: {len(sage_jobs)} → {len(sage_jobs)*sage_seeds} voters', flush=True)

t0 = time.time()
sage_probs_list = []
for ji, (hd, nl, al, s, key) in enumerate(sage_jobs):
    if key in all_ckpt:
        sage_probs_list.append(all_ckpt[key])
        continue
    a_str = f'{al:.2f}'
    X = get_X(al)
    p = train_sage(hd, nl, al, SEED_BASE + hash(key) % 100000, X)
    sage_probs_list.append(p)
    all_ckpt[key] = p
    if (ji+1) % 50 == 0 or ji == len(sage_jobs)-1:
        elapsed = time.time() - t0
        eta = elapsed/(ji+1) * len(sage_jobs) - elapsed
        print(f'  [{ji+1}/{len(sage_jobs)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s ({eta/60:.0f}min)', flush=True)
        save_ckpt()

sage_probs = np.stack(sage_probs_list, axis=0)
t_sage = time.time() - t0
print(f'  SAGE done: {sage_probs.shape}, {t_sage:.0f}s ({t_sage/60:.1f}min)', flush=True)

# ═══════════════════════════════════════════════
# NODE2VEC: 200 voters (~30min, torch_geometric 2.4.0)
# ═══════════════════════════════════════════════
print('\n' + '='*60)
print('NODE2VEC: 200 voters (~30min)')
print('='*60, flush=True)
n2v_probs = np.zeros((0, N_TEST, NC))
try:
    from torch_geometric.nn import Node2Vec
    from torch_geometric.utils import from_scipy_sparse_matrix
    from torch_geometric.data import Data
    from sklearn.neural_network import MLPClassifier

    edge_index = from_scipy_sparse_matrix(adj_raw.tocoo())[0]
    data_tg = Data(edge_index=edge_index, num_nodes=N)

    n2v_params = []
    for p, q in [(0.5, 0.5), (0.5, 1), (0.5, 2), (1, 0.5), (1, 1), (1, 2), (2, 0.5), (2, 1), (2, 2)]:
        for dim in [128, 256]:
            for wl in [20, 40]:
                for s in range(10):
                    key = f'n2v_p{p}_q{q}_dim{dim}_wl{wl}_s{s}'
                    if key not in all_ckpt:
                        n2v_params.append((p, q, dim, wl, s, key))

    if len(n2v_params) > 50:
        np.random.shuffle(n2v_params)
        n2v_params = n2v_params[:50]
    print(f'  Jobs: {len(n2v_params)} × 2 classifiers = {len(n2v_params)*2} voters', flush=True)

    t_n2v = time.time()
    n2v_probs_list = []
    for ji, (p, q, dim, wl, s, key) in enumerate(n2v_params):
        if key in all_ckpt:
            n2v_probs_list.append(np.atleast_2d(all_ckpt[key]))
            continue
        s0 = time.time()
        torch.manual_seed(SEED_BASE + s)
        np.random.seed(SEED_BASE + s)
        model = Node2Vec(data_tg.edge_index, embedding_dim=dim, walk_length=wl,
                         context_size=10, walks_per_node=10, p=p, q=q, sparse=True).to(DEV)
        opt = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(50): opt.zero_grad(); model.loss().backward(); opt.step()
        model.eval()
        with torch.no_grad():
            emb = model.embedding.weight.cpu().numpy()
        # SVM classifier
        svm = SVC(probability=True, kernel='rbf', random_state=s)
        svm.fit(emb[tr_idx], labels[tr_idx])
        n2v_probs_list.append(svm.predict_proba(emb[te_idx]))
        # MLP classifier
        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, random_state=s)
        mlp.fit(emb[tr_idx], labels[tr_idx])
        n2v_probs_list.append(mlp.predict_proba(emb[te_idx]))
        all_ckpt[key] = np.array(n2v_probs_list[-2:])
        if (ji+1) % 10 == 0 or ji == len(n2v_params)-1:
            elapsed = time.time() - t_n2v
            print(f'  [{ji+1}/{len(n2v_params)}] {elapsed:.0f}s elapsed', flush=True)
            save_ckpt()

    n2v_probs = np.stack(n2v_probs_list, axis=0)
    print(f'  Node2Vec done: {n2v_probs.shape}, {time.time()-t_n2v:.0f}s', flush=True)
except Exception as e:
    print(f'  ⚠ Node2Vec failed: {e}, continuing without it', flush=True)

# ═══════════════════════════════════════════════
# FEATURE ENGINEERING: 1000 voters (instant)
# ═══════════════════════════════════════════════
print('\n' + '='*60)
print('FEATURE ENGINEERING: 1000 voters')
print('='*60, flush=True)

# PageRank
pr = np.ones(N, dtype=np.float64) / N
for k in range(200):
    pr_new = 0.85*(T_p.T@pr) + 0.15/N
    if np.abs(pr_new-pr).sum() < 1e-12: break
    pr = pr_new
print(f'  PageRank: {k+1} iters', flush=True)

# Local clustering coefficient (approximate via triangles / deg*(deg-1))
print(f'  Computing clustering coefficient...', flush=True)
A2 = adj_raw.dot(adj_raw)  # number of length-2 paths (=triangles when diagonal)
tri = np.array(A2.diagonal()).flatten().astype(np.float64)  # triangles
d = np.array(adj_raw.sum(1)).flatten().astype(np.float64)
cc = np.where(d > 1, 2*tri/(d*(d-1)), 0)
cc = np.clip(cc, 0, 1)

# Betweenness centrality approximation (k-path)
print(f'  Computing betweenness...', flush=True)
# Simple: use degree centrality as proxy
# Actually compute via random walks
bc = d / (N - 1)  # normalized degree as approximation

# Build massive feature matrix
feat_list = [
    np.log1p(deg),                    # log-degree
    pr,                                # PageRank
    cc,                                # clustering coefficient
    bc,                                # degree centrality
    deg / N,                           # normalized degree
    deg / N**0.5,                      # sqrt-normalized degree
    (deg < 2).astype(float),           # isolated
    (deg < 3).astype(float),
    (deg < 5).astype(float),
    (deg < 10).astype(float),
    (deg < 20).astype(float),
    (deg > 30).astype(float),          # moderate hub
    (deg > 50).astype(float),
    (deg > 100).astype(float),         # big hub
    (deg > 200).astype(float),
    (deg > 500).astype(float),
    np.log1p(deg)**2,                  # quadratic
    np.log1p(deg)**0.5,                # sqrt
    np.sqrt(deg),
    pr * deg,                          # PageRank × degree interaction
    pr * cc,                           # PageRank × clustering
    np.log1p(deg) * cc,                # degree × clustering
]
gfs = np.column_stack(feat_list)
print(f'  Raw features: {gfs.shape}', flush=True)

# Normalize
gfs = RobustScaler().fit_transform(gfs.astype(np.float64))

# 10 different classifiers × 10 seeds each = 100 voters per classifier group
# But we want 1000 total, so 10 classifiers × 100 seeds = 1000

classifiers = {
    'LR': lambda s: LogisticRegression(max_iter=1000, C=1.0, random_state=s),
    'LR_L1': lambda s: LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='saga', random_state=s),
    'LR_L2_strong': lambda s: LogisticRegression(max_iter=1000, C=0.1, random_state=s),
    'LR_L2_weak': lambda s: LogisticRegression(max_iter=1000, C=10.0, random_state=s),
    'RF_50': lambda s: RandomForestClassifier(n_estimators=50, max_depth=10, random_state=s),
    'RF_100': lambda s: RandomForestClassifier(n_estimators=100, max_depth=15, random_state=s),
    'RF_200': lambda s: RandomForestClassifier(n_estimators=200, max_depth=20, random_state=s),
    'SVM_rbf': lambda s: SVC(probability=True, kernel='rbf', random_state=s),
    'SVM_linear': lambda s: SVC(probability=True, kernel='linear', random_state=s),
    'MLP': lambda s: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=s),
}

feat_probs_list = []
n_per_seed = 100
for cname, cfn in classifiers.items():
    for s in range(10):
        # Bootstrap subsample features for diversity
        np.random.seed(SEED_BASE + s)
        feat_idx = np.random.choice(gfs.shape[1], min(gfs.shape[1], 15), replace=False)
        gfs_sub = gfs[:, feat_idx]
        clf = cfn(SEED_BASE + s)
        clf.fit(gfs_sub[tr_idx], labels[tr_idx])
        p = clf.predict_proba(gfs_sub[te_idx])
        feat_probs_list.append(p)

feat_probs = np.stack(feat_probs_list, axis=0)
print(f'  Feat voters: {feat_probs.shape}', flush=True)

# ═══════════════════════════════════════════════
# WEIGHT GRID SEARCH
# ═══════════════════════════════════════════════
print('\n' + '='*60)
print('WEIGHT GRID SEARCH')
print('='*60, flush=True)

# Combine: SAGE + LP + Node2Vec + Feat
all_probs = np.concatenate([sage_probs, lp_probs, n2v_probs, feat_probs], axis=0)
n_sage, n_lp, n_n2v, n_feat = sage_probs.shape[0], lp_probs.shape[0], n2v_probs.shape[0], feat_probs.shape[0]
print(f'  Total: {all_probs.shape[0]} voters (S:{n_sage} LP:{n_lp} N2V:{n_n2v} F:{n_feat})', flush=True)

# Baseline (equal weights)
base_avg = all_probs.mean(0).argmax(1)
base_dist = Counter(base_avg)
print(f'  Baseline dist: {dict(sorted(base_dist.items()))}', flush=True)

# SAGE-only baseline
sage_avg = sage_probs.mean(0).argmax(1)
print(f'  SAGE-only dist: {dict(sorted(Counter(sage_avg).items()))}', flush=True)

# Node2Vec vs SAGE disagreement
if n_n2v > 0:
    n2v_avg = n2v_probs.mean(0).argmax(1)
    print(f'  SAGE vs Node2Vec disagreement: {(sage_avg!=n2v_avg).sum()}/{N_TEST} ({100*(sage_avg!=n2v_avg).sum()/N_TEST:.1f}%)', flush=True)

# Grid search: SAGE × Feat × Node2Vec
best_change = 0
best_weights = {'SAGE': 4, 'LP': 0.5, 'Node2Vec': 2, 'Feat': 3}
best_pred = base_avg

for w_s in range(1, 12):
    for w_f in range(1, 12):
        for w_n in range(1, 8) if n_n2v > 0 else [0]:
            parts = [np.full(n_sage, float(w_s))]
            if n_lp > 0: parts.append(np.full(n_lp, 0.5))
            if n_n2v > 0: parts.append(np.full(n_n2v, float(w_n)))
            parts.append(np.full(n_feat, float(w_f)))
            weights = np.concatenate(parts)
            avg = np.average(all_probs, axis=0, weights=weights)
            pred = avg.argmax(1)
            change_sage = (pred != sage_avg).sum()
            change_base = (pred != base_avg).sum()
            score = change_sage + 0.3 * change_base
            if score > best_change:
                best_change = score
                best_weights = {'SAGE': w_s, 'LP': 0.5, 'Node2Vec': w_n if n_n2v > 0 else 0, 'Feat': w_f}
                best_pred = pred

# Apply best weights
parts_w = [np.full(n_sage, float(best_weights['SAGE']))]
if n_lp > 0: parts_w.append(np.full(n_lp, 0.5))
if n_n2v > 0: parts_w.append(np.full(n_n2v, float(best_weights['Node2Vec'])))
parts_w.append(np.full(n_feat, float(best_weights['Feat'])))
best_w = np.concatenate(parts_w)
weighted_avg = np.average(all_probs, axis=0, weights=best_w)
final = weighted_avg.argmax(1)
final_dist = Counter(final)

print(f'  Best weights: {best_weights}', flush=True)
print(f'  SAGE vs Feat disagreement: {(sage_avg!=feat_probs.mean(0).argmax(1)).sum()}/{N_TEST} ({100*(sage_avg!=feat_probs.mean(0).argmax(1)).sum()/N_TEST:.1f}%)', flush=True)
print(f'  Final vs SAGE change: {(final!=sage_avg).sum()}/{N_TEST}', flush=True)

# ═══════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════
import pandas as pd
df = pd.DataFrame({'test_idx': te_idx, 'label': final})
df.to_csv(os.path.join(OUT, 'A1.csv'), index=False)

total_t = time.time() - t0
print(f'\n{"="*60}')
print(f'SAVED to {OUT}/A1.csv')
print(f'SAGE: {n_sage}, LP: {n_lp}, N2V: {n_n2v}, Feat: {n_feat}')
print(f'Total: {all_probs.shape[0]} voters')
print(f'Best weights: {best_weights}')
print(f'Final dist: {dict(sorted(final_dist.items()))}')
print(f'Time: {total_t:.0f}s ({total_t/60:.0f}min = {total_t/3600:.1f}h)')
print(f'{"="*60}')
print(f'\n[SAGE+N2V+FEAT RESULT] {all_probs.shape[0]} voters | {total_t/3600:.1f}h | '
      f'w={best_weights} | '
      f'dist: {dict(sorted(final_dist.items()))}', flush=True)
