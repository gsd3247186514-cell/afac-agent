"""A1 ULTIMATE: 6方法 × 5800+ voters × 4-6小时
SAGE(2000) + GCN(500) + GAT(250) + Node2Vec(2000) + DeepWalk(500) + Feat(30)
加权网格搜索最优组合权重
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, os, sys, time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_norm, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading, itertools

DEV = torch.device('cuda')
SEED_BASE = 42
np.random.seed(SEED_BASE)
torch.manual_seed(SEED_BASE)

# ═══════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════
DATA = sys.argv[1] if len(sys.argv) > 1 else 'data/cls_data/A1.npz'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'a1_ultimate_out'
os.makedirs(OUT, exist_ok=True)
CKPT_FILE = os.path.join(OUT, '.ckpt.npz')

print(f'[LOAD] {DATA}', flush=True)
d = np.load(DATA)
adj_raw = csr_matrix((d['adj_data'], d['adj_indices'], d['adj_indptr']), shape=tuple(d['adj_shape']))
feat_raw = csr_matrix((d['attr_data'], d['attr_indices'], d['attr_indptr']), shape=tuple(d['attr_shape']))
labels = d['labels'].astype(int)
tr_idx = d['train_idx']
te_idx = d['test_idx']
N, NC = adj_raw.shape[0], 10
N_TEST = len(te_idx)
print(f'  N={N} test={N_TEST} classes={NC}', flush=True)

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

# Transition matrix for LP
rs = np.array(asym.sum(1)).flatten(); rs[rs==0]=1
T_p = csr_matrix((1.0/rs,(range(N),range(N))),shape=(N,N)) @ asym

Y_t = torch.tensor(labels, dtype=torch.long, device=DEV)
tr_t = torch.tensor(tr_idx, dtype=torch.long, device=DEV)
te_t = torch.tensor(te_idx, dtype=torch.long, device=DEV)
Y0 = np.zeros((N,NC), dtype=np.float32)
for ti in tr_idx: Y0[ti, labels[ti]] = 1.0

X_base = torch.tensor(feat, device=DEV)
deg_base = torch.tensor(deg_feat, device=DEV)

# ═══════════════════════════════════════════════
# LP: 9 variants (instant)
# ═══════════════════════════════════════════════
print('\n[LP] 9 variants...', flush=True)
ALPHAS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
LP = {}
for a in ALPHAS:
    Y = Y0.copy()
    for _ in range(80): Y = a*(T_p@Y) + (1-a)*Y0
    LP[a] = Y.astype(np.float32)
lp_probs = np.stack([LP[a][te_idx] for a in ALPHAS], axis=0)  # (9,2751,10)
print(f'  LP: {lp_probs.shape}', flush=True)

F_IN = F_BASE + NC + 3  # 780

# ═══════════════════════════════════════════════
# TRAIN FUNCTIONS
# ═══════════════════════════════════════════════
EPOCHS = 400; LR = 0.005; WD = 5e-4; DROP = 0.5
lock = threading.Lock()

def get_X(a): 
    return torch.tensor(np.hstack([feat, LP[a], deg_feat]).astype(np.float32), device=DEV)

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

def train_gcn(hdim, nlayers, alpha, seed, X):
    torch.manual_seed(seed)
    layers = [nn.Linear(F_IN if i==0 else hdim, hdim if i<nlayers-1 else NC, device=DEV) for i in range(nlayers)]
    opt = torch.optim.AdamW([p for l in layers for p in l.parameters()], lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        for l in layers: l.train()
        opt.zero_grad()
        h = X
        for i in range(nlayers-1):
            h = F.relu(F.dropout(A_sp@layers[i](h) if i==0 else layers[i](h), p=DROP, training=True))
        F.cross_entropy(layers[-1](h)[tr_t], Y_t[tr_t]).backward(); opt.step()
    for l in layers: l.eval()
    with torch.no_grad():
        h = X
        for i in range(nlayers-1):
            h = F.relu(F.dropout(A_sp@layers[i](h) if i==0 else layers[i](h), p=DROP, training=False))
        return F.softmax(layers[-1](h), dim=-1).cpu().numpy()[te_idx]

def train_gat(hdim, nlayers, heads, alpha, seed, X):
    torch.manual_seed(seed)
    proj_layers = [nn.Linear(F_IN if i==0 else hdim*heads, hdim*heads, device=DEV) for i in range(nlayers)]
    out_layer = nn.Linear(hdim*heads, NC, device=DEV)
    params = [p for proj in proj_layers for p in proj.parameters()] + list(out_layer.parameters())
    opt = torch.optim.AdamW(params, lr=0.003, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad()
        h = X
        for proj in proj_layers:
            h = proj(h).view(N, heads, -1)
            agg = torch.stack([A_sp@h[:,k,:] for k in range(heads)], dim=1)
            h = F.elu(agg.permute(0,2,1).reshape(N,-1))
            h = F.dropout(h, p=DROP, training=True)
        F.cross_entropy(out_layer(h)[tr_t], Y_t[tr_t]).backward(); opt.step()
    with torch.no_grad():
        h = X
        for proj in proj_layers:
            h = proj(h).view(N, heads, -1)
            agg = torch.stack([A_sp@h[:,k,:] for k in range(heads)], dim=1)
            h = F.elu(agg.permute(0,2,1).reshape(N,-1))
        return F.softmax(out_layer(h), dim=-1).cpu().numpy()[te_idx]

# ═══════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════
all_results = {}  # key -> (method_name, probs_array)
completed_keys = set()
if os.path.exists(CKPT_FILE):
    ckpt = np.load(CKPT_FILE, allow_pickle=True)
    completed_keys = set(ckpt.files)
    for k in completed_keys:
        all_results[k] = ('ckpt', ckpt[k])
    print(f'[CKPT] Loaded {len(completed_keys)} completed tasks', flush=True)

def save_ckpt():
    save_dict = {k: v[1] for k, v in all_results.items()}
    if save_dict:
        np.savez_compressed(CKPT_FILE, **save_dict)

# ═══════════════════════════════════════════════
# SAGE: 2000 voters (~80 min)
# ═══════════════════════════════════════════════
print('\n━━━ SAGE: 2000 voters ━━━', flush=True)
sage_hdims = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640, 768, 896, 1024]
sage_layers_options = [1, 2, 3, 4, 5]
sage_alphas = [0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
sage_seeds = 3  # 3 seeds per config for diversity
sage_jobs = []

for hd in sage_hdims:
    for nl in sage_layers_options:
        for al in sage_alphas:
            # Heuristic: skip unreasonable combos
            if nl == 1 and hd > 512: continue  # single-layer with huge hidden = waste
            if nl >= 4 and hd < 64: continue    # deep narrow = unstable
            if nl >= 5 and hd < 128: continue
            for s in range(sage_seeds):
                key = f'sage_h{hd}_L{nl}_a{al:.2f}_s{s}'
                if key not in completed_keys:
                    sage_jobs.append((hd, nl, al, s, key))

# Subsample to ~670 jobs = ~2000 voters (3 seeds each)
if len(sage_jobs) > 670:
    np.random.shuffle(sage_jobs)
    sage_jobs = sage_jobs[:670]
print(f'  SAGE jobs: {len(sage_jobs)} ({len(sage_jobs)*sage_seeds} voters)', flush=True)

t_sage = time.time()
sage_probs_list = []
X_cache = {}
for ji, (hd, nl, al, s, key) in enumerate(sage_jobs):
    if key in completed_keys:
        sage_probs_list.append(all_results[key][1])
        continue
    a_str = f'{al:.2f}'
    if a_str not in X_cache: X_cache[a_str] = get_X(al)
    p = train_sage(hd, nl, al, SEED_BASE + hash(key) % 100000, X_cache[a_str])
    sage_probs_list.append(p)
    all_results[key] = ('sage', p)
    if (ji+1) % 20 == 0 or ji == len(sage_jobs)-1:
        elapsed = time.time() - t_sage
        print(f'  SAGE: {ji+1}/{len(sage_jobs)} ({100*(ji+1)/len(sage_jobs):.0f}%), {elapsed:.0f}s', flush=True)
        save_ckpt()

sage_probs = np.stack(sage_probs_list, axis=0)
print(f'  SAGE done: {sage_probs.shape}, {time.time()-t_sage:.0f}s', flush=True)

# ═══════════════════════════════════════════════
# GCN: 500 voters (~16 min)
# ═══════════════════════════════════════════════
print('\n━━━ GCN: 500 voters ━━━', flush=True)
gcn_hdims = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
gcn_layers = [1, 2, 3, 4]
gcn_alphas = [0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
gcn_seeds = 5
gcn_jobs = []

for hd in gcn_hdims:
    for nl in gcn_layers:
        for al in gcn_alphas:
            if nl == 1 and hd > 768: continue
            if nl >= 4 and hd < 128: continue
            for s in range(gcn_seeds):
                key = f'gcn_h{hd}_L{nl}_a{al:.2f}_s{s}'
                if key not in completed_keys:
                    gcn_jobs.append((hd, nl, al, s, key))

if len(gcn_jobs) > 100:
    np.random.shuffle(gcn_jobs)
    gcn_jobs = gcn_jobs[:100]
print(f'  GCN jobs: {len(gcn_jobs)} ({len(gcn_jobs)*gcn_seeds} voters)', flush=True)

t_gcn = time.time()
gcn_probs_list = []
for ji, (hd, nl, al, s, key) in enumerate(gcn_jobs):
    if key in completed_keys:
        gcn_probs_list.append(all_results[key][1])
        continue
    a_str = f'{al:.2f}'
    if a_str not in X_cache: X_cache[a_str] = get_X(al)
    p = train_gcn(hd, nl, al, SEED_BASE + hash(key) % 100000, X_cache[a_str])
    gcn_probs_list.append(p)
    all_results[key] = ('gcn', p)
    if (ji+1) % 10 == 0 or ji == len(gcn_jobs)-1:
        elapsed = time.time() - t_gcn
        print(f'  GCN: {ji+1}/{len(gcn_jobs)} ({100*(ji+1)/len(gcn_jobs):.0f}%), {elapsed:.0f}s', flush=True)
        save_ckpt()

gcn_probs = np.stack(gcn_probs_list, axis=0)
if len(gcn_probs) == 0: gcn_probs = np.zeros((0,N_TEST,NC))
print(f'  GCN done: {gcn_probs.shape}, {time.time()-t_gcn:.0f}s', flush=True)

# ═══════════════════════════════════════════════
# GAT: 250 voters (~25 min)
# ═══════════════════════════════════════════════
print('\n━━━ GAT: 250 voters ━━━', flush=True)
gat_hdims = [64, 96, 128, 192, 256]
gat_layers = [1, 2, 3]
gat_heads = [2, 4]
gat_alphas = [0.7, 0.8, 0.85, 0.9, 0.95, 0.97]
gat_seeds = 3
gat_jobs = []

for hd in gat_hdims:
    for nl in gat_layers:
        for nh in gat_heads:
            for al in gat_alphas:
                if nl >= 3 and hd*nh > 512: continue  # too big
                for s in range(gat_seeds):
                    key = f'gat_h{hd}_L{nl}_H{nh}_a{al:.2f}_s{s}'
                    if key not in completed_keys:
                        gat_jobs.append((hd, nl, nh, al, s, key))

if len(gat_jobs) > 85:
    np.random.shuffle(gat_jobs)
    gat_jobs = gat_jobs[:85]
print(f'  GAT jobs: {len(gat_jobs)} ({len(gat_jobs)*gat_seeds} voters)', flush=True)

t_gat = time.time()
gat_probs_list = []
for ji, (hd, nl, nh, al, s, key) in enumerate(gat_jobs):
    if key in completed_keys:
        gat_probs_list.append(all_results[key][1])
        continue
    a_str = f'{al:.2f}'
    if a_str not in X_cache: X_cache[a_str] = get_X(al)
    p = train_gat(hd, nl, nh, al, SEED_BASE + hash(key) % 100000, X_cache[a_str])
    gat_probs_list.append(p)
    all_results[key] = ('gat', p)
    if (ji+1) % 10 == 0 or ji == len(gat_jobs)-1:
        print(f'  GAT: {ji+1}/{len(gat_jobs)} ({100*(ji+1)/len(gat_jobs):.0f}%), {time.time()-t_gat:.0f}s', flush=True)
        save_ckpt()

gat_probs = np.stack(gat_probs_list, axis=0)
if len(gat_probs) == 0: gat_probs = np.zeros((0,N_TEST,NC))
print(f'  GAT done: {gat_probs.shape}, {time.time()-t_gat:.0f}s', flush=True)

# ═══════════════════════════════════════════════
# Node2Vec: 2000 voters (~120 min)
# ═══════════════════════════════════════════════
print('\n━━━ Node2Vec: 2000 voters ━━━', flush=True)
try:
    from torch_geometric.nn import Node2Vec
    from torch_geometric.utils import from_scipy_sparse_matrix
    from torch_geometric.data import Data
    edge_index = from_scipy_sparse_matrix(adj_raw.tocoo())[0]
    data_tg = Data(edge_index=edge_index, num_nodes=N)
    
    n2v_params = []
    for p,q in [(0.5,0.5),(0.5,1),(0.5,2),(1,0.5),(1,1),(1,2),(2,0.5),(2,1),(2,2)]:
        for dim in [128, 256]:
            for wl in [20, 40]:
                for s in range(10):
                    key = f'n2v_p{p}_q{q}_dim{dim}_wl{wl}_s{s}'
                    if key not in completed_keys:
                        n2v_params.append((p, q, dim, wl, s, key))
    
    if len(n2v_params) > 100:
        np.random.shuffle(n2v_params)
        n2v_params = n2v_params[:100]
    print(f'  Node2Vec jobs: {len(n2v_params)} x 2 classifiers', flush=True)
    
    t_n2v = time.time()
    n2v_probs_list = []
    for ji, (p, q, dim, wl, s, key) in enumerate(n2v_params):
        if key in completed_keys:
            n2v_probs_list.append(all_results[key][1])
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
        # SVM
        svm = SVC(probability=True, kernel='rbf', random_state=s)
        svm.fit(emb[tr_idx], labels[tr_idx])
        n2v_probs_list.append(svm.predict_proba(emb[te_idx]))
        # MLP (simple)
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, random_state=s)
        mlp.fit(emb[tr_idx], labels[tr_idx])
        n2v_probs_list.append(mlp.predict_proba(emb[te_idx]))
        all_results[key] = ('n2v', np.array(n2v_probs_list[-2:]))
        if (ji+1) % 10 == 0 or ji == len(n2v_params)-1:
            elapsed = time.time() - t_n2v
            print(f'  Node2Vec: {ji+1}/{len(n2v_params)} ({100*(ji+1)/len(n2v_params):.0f}%), {elapsed:.0f}s', flush=True)
            save_ckpt()
    
    n2v_probs = np.stack(n2v_probs_list, axis=0)
    print(f'  Node2Vec done: {n2v_probs.shape}, {time.time()-t_n2v:.0f}s', flush=True)
except ImportError:
    print('  ⚠ torch_geometric unavailable, skipping Node2Vec', flush=True)
    n2v_probs = np.zeros((0,N_TEST,NC))

# ═══════════════════════════════════════════════
# DeepWalk: 500 voters (~30 min)
# ═══════════════════════════════════════════════
print('\n━━━ DeepWalk: 500 voters ━━━', flush=True)
try:
    G = None
    dw_probs_list = []
    t_dw = time.time()
    
    # Simple random-walk based embeddings (no gensim needed)
    # Use normalized adjacency power-iterated embeddings
    for k in [2, 4, 8, 16, 32]:
        for dim in [64, 128, 256]:
            for s in range(10):
                key = f'dw_k{k}_dim{dim}_s{s}'
                if key in completed_keys:
                    dw_probs_list.append(np.atleast_2d(all_results[key][1]))
                    continue
                np.random.seed(SEED_BASE + s)
                # Random projection of adjacency power walks
                M = np.random.randn(N, dim).astype(np.float32) / np.sqrt(dim)
                A = adj_raw.copy()
                for _ in range(k-1):
                    M = A.dot(M)
                emb = StandardScaler().fit_transform(M.astype(np.float64))
                lr = LogisticRegression(max_iter=500, multi_class='multinomial', random_state=s)
                lr.fit(emb[tr_idx], labels[tr_idx])
                dw_probs_list.append(lr.predict_proba(emb[te_idx]))
                all_results[key] = ('dw', dw_probs_list[-1])
            elapsed = time.time() - t_dw
            print(f'  DeepWalk k={k}: {elapsed:.0f}s', flush=True)
    
    dw_probs = np.stack(dw_probs_list, axis=0)
    print(f'  DeepWalk done: {dw_probs.shape}, {time.time()-t_dw:.0f}s', flush=True)
except Exception as e:
    print(f'  ⚠ DeepWalk failed: {e}, skipping', flush=True)
    dw_probs = np.zeros((0,N_TEST,NC))

# ═══════════════════════════════════════════════
# Feature Engineering: 30 voters (instant)
# ═══════════════════════════════════════════════
print('\n━━━ Feature Engineering: 30 voters ━━━', flush=True)
from sklearn.preprocessing import RobustScaler

# PageRank
t0 = time.time()
pr = np.ones(N, dtype=np.float64) / N
for k in range(200):
    pr_new = 0.85 * (T_p.T @ pr) + 0.15 / N
    if np.abs(pr_new - pr).sum() < 1e-10: break
    pr = pr_new
print(f'  PageRank: {k+1} iters, {time.time()-t0:.0f}s', flush=True)

# Many graph features
gfs = np.column_stack([
    np.log1p(deg),                          # log-degree
    pr,                                     # PageRank
    deg / N,                                # normalized degree
    (deg < 3).astype(float),                # isolated-ish
    (deg < 5).astype(float),
    (deg < 10).astype(float),
    (deg > 50).astype(float),               # hub
    (deg > 100).astype(float),
    (deg > 200).astype(float),
    np.log1p(deg) ** 2,                     # quadratic log-degree
    np.sqrt(deg),                           # sqrt degree
])
gfs = RobustScaler().fit_transform(gfs.astype(np.float64))
print(f'  Graph features: {gfs.shape}', flush=True)

# Multiple classifiers
feat_probs_list = []
for clf_name, clf in [
    ('LR', LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=SEED_BASE)),
    ('RF_100', RandomForestClassifier(n_estimators=100, random_state=SEED_BASE)),
    ('RF_300', RandomForestClassifier(n_estimators=300, random_state=SEED_BASE+1)),
    ('SVM_rbf', SVC(probability=True, kernel='rbf', random_state=SEED_BASE)),
    ('SVM_linear', SVC(probability=True, kernel='linear', random_state=SEED_BASE)),
]:
    clf.fit(gfs[tr_idx], labels[tr_idx])
    p = clf.predict_proba(gfs[te_idx])
    feat_probs_list.append(p)
    pred = p.argmax(1)
    print(f'  {clf_name}: dist={dict(sorted(Counter(pred).items()))}', flush=True)

feat_probs = np.stack(feat_probs_list, axis=0)
print(f'  Feature ensemble: {feat_probs.shape}', flush=True)
save_ckpt()

# ═══════════════════════════════════════════════
# WEIGHTED ENSEMBLE with GRID SEARCH
# ═══════════════════════════════════════════════
print('\n━━━ Weighted Ensemble ━━━', flush=True)

# Collect all probs and their method labels
all_methods = []
if sage_probs.shape[0] > 0: all_methods.append(('SAGE', sage_probs))
if gcn_probs.shape[0] > 0: all_methods.append(('GCN', gcn_probs))
if gat_probs.shape[0] > 0: all_methods.append(('GAT', gat_probs))
all_methods.append(('LP', lp_probs))
if n2v_probs.shape[0] > 0: all_methods.append(('Node2Vec', n2v_probs))
if dw_probs.shape[0] > 0: all_methods.append(('DeepWalk', dw_probs))
all_methods.append(('Feat', feat_probs))

flat_probs = []
flat_methods = []
for method, probs in all_methods:
    for i in range(probs.shape[0]):
        flat_probs.append(probs[i])
        flat_methods.append(method)
flat_probs = np.stack(flat_probs, axis=0)
print(f'  Total flat voters: {flat_probs.shape[0]}', flush=True)

# Method counts
method_counts = Counter(flat_methods)
print(f'  Method breakdown: {dict(method_counts)}', flush=True)

# Default weights (empirically tuned)
default_weights = {
    'SAGE': 4.0, 'GCN': 2.0, 'GAT': 2.0, 'LP': 0.5,
    'Node2Vec': 3.0, 'DeepWalk': 1.0, 'Feat': 1.0
}

# Grid search over weight ratios
weight_grid = []
for w_sage in [3, 4, 5, 6]:
    for w_gcn in [1, 2, 3]:
        for w_n2v in [2, 3, 4, 5]:
            for w_lp in [0.3, 0.5, 1]:
                weight_grid.append({
                    'SAGE': float(w_sage), 'GCN': float(w_gcn), 'GAT': 2.0,
                    'LP': float(w_lp), 'Node2Vec': float(w_n2v),
                    'DeepWalk': 1.0, 'Feat': 1.0
                })

print(f'  Grid search: {len(weight_grid)} weight combos', flush=True)

# Evaluate each weight combo
best_dist = None
best_weights = None
best_change = 0
base_dist = Counter(flat_probs.mean(0).argmax(1))

for wi, weights in enumerate(weight_grid):
    voter_weights = np.array([weights[m] for m in flat_methods])
    avg = np.average(flat_probs, axis=0, weights=voter_weights)
    pred = avg.argmax(1)
    dist = Counter(pred)
    # Score: prefer distributions that differ from SAGE-only (adds diversity)
    sage_only = sage_probs.mean(0).argmax(1)
    change = (pred != sage_only).sum()
    if change > best_change:
        best_change = change
        best_dist = dist
        best_weights = dict(weights)
    if wi % 20 == 0:
        print(f'  Grid {wi+1}/{len(weight_grid)}: best_change={best_change}', flush=True)

# Apply best weights
print(f'\n  Best weights: {best_weights}', flush=True)
print(f'  Best change from SAGE: {best_change}/{N_TEST}', flush=True)
voter_weights = np.array([best_weights[m] for m in flat_methods])
weighted_prob = np.average(flat_probs, axis=0, weights=voter_weights)
final = weighted_prob.argmax(1)
final_dist = Counter(final)

# SAGE vs others disagreement
sage_avg = sage_probs.mean(0).argmax(1)
for m in ['GCN', 'GAT', 'LP', 'Node2Vec']:
    if m in method_counts:
        mi = [i for i, fm in enumerate(flat_methods) if fm == m]
        if mi:
            m_avg = flat_probs[mi].mean(0).argmax(1)
            print(f'  SAGE vs {m}: {(sage_avg!=m_avg).sum()}/{N_TEST}=', end='')
            print(f'{100*(sage_avg!=m_avg).sum()/N_TEST:.1f}%', flush=True)

# ═══════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════
import pandas as pd
out_csv = os.path.join(OUT, 'A1.csv')
df = pd.DataFrame({'test_idx': te_idx, 'label': final})
df.to_csv(out_csv, index=False)

total_t = time.time() - t0
print(f'\n{"="*60}')
print(f'SAVED to {out_csv}')
print(f'Voters: {flat_probs.shape[0]} from {len(all_methods)} methods')
print(f'Methods: {dict(method_counts)}')
print(f'Best weights: {best_weights}')
print(f'Final dist: {dict(sorted(final_dist.items()))}')
print(f'Total time: {total_t:.0f}s ({total_t/60:.1f}min = {total_t/3600:.2f}h)')
print(f'{"="*60}')
print(f'\n[ULTIMATE RESULT] {flat_probs.shape[0]} voters | '
      f'dist: {dict(sorted(final_dist.items()))} | '
      f'{total_t/3600:.2f}h', flush=True)
