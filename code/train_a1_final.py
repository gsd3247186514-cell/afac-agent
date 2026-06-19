"""A1 FINAL: SAGE + LP + Node2Vec + 特征工程 + 加权投票
目标: 0.77, 运行时间: ~30min (V100)
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, os, sys, time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_norm, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading

DEV = torch.device('cuda')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ═══════════════════════════════════════════════
# PART 0: 加载数据
# ═══════════════════════════════════════════════
DATA = sys.argv[1] if len(sys.argv) > 1 else 'data/cls_data/A1.npz'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'a1_final_out'
os.makedirs(OUT, exist_ok=True)

print(f'[0/6] Loading {DATA}...', flush=True)
d = np.load(DATA)
adj_raw = csr_matrix((d['adj_data'], d['adj_indices'], d['adj_indptr']), shape=tuple(d['adj_shape']))
feat_raw = csr_matrix((d['attr_data'], d['attr_indices'], d['attr_indptr']), shape=tuple(d['attr_shape']))
labels = d['labels'].astype(int)
tr_idx = d['train_idx']
te_idx = d['test_idx']
N = adj_raw.shape[0]
NC = 10
N_TEST = len(te_idx)
print(f'  N={N}, test={N_TEST}, classes={NC}', flush=True)

# 特征预处理
feat = sk_norm(feat_raw.toarray().astype(np.float32), norm='l2', axis=1)
feat[np.isnan(feat).any(axis=1)] = 0
F_BASE = feat.shape[1]

deg = np.array(adj_raw.sum(1)).flatten().astype(np.float32)
deg_feat = np.hstack([np.log1p(deg).reshape(-1, 1),
    (deg < 5).astype(np.float32).reshape(-1, 1), (deg < 10).astype(np.float32).reshape(-1, 1)])

# 对称归一化邻接矩阵
asym = adj_raw + adj_raw.T
asym.setdiag(1)
ds = np.array(asym.sum(1)).flatten()
dis = np.where(ds > 0, 1.0 / np.sqrt(ds), 0)
An = csr_matrix((dis, (range(N), range(N))), shape=(N, N)) @ asym @ csr_matrix((dis, (range(N), range(N))), shape=(N, N))
coo = An.tocoo()
A_sp = torch.sparse_coo_tensor(
    torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=DEV),
    torch.tensor(coo.data, dtype=torch.float32, device=DEV), (N, N)).coalesce()

# 转移矩阵 (LP用)
rs = np.array(asym.sum(1)).flatten()
rs[rs == 0] = 1
T_p = csr_matrix((1.0 / rs, (range(N), range(N))), shape=(N, N)) @ asym

Y_t = torch.tensor(labels, dtype=torch.long, device=DEV)
tr_t = torch.tensor(tr_idx, dtype=torch.long, device=DEV)
te_t = torch.tensor(te_idx, dtype=torch.long, device=DEV)

Y0 = np.zeros((N, NC), dtype=np.float32)
for ti in tr_idx:
    Y0[ti, labels[ti]] = 1.0

# ═══════════════════════════════════════════════
# PART 1: LP计算 (9种α)
# ═══════════════════════════════════════════════
print('\n[1/6] Computing LP (9 alphas)...', flush=True)
ALPHAS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
LP = {}

def run_lp(alpha, iters=80):
    Y = Y0.copy()
    for _ in range(iters):
        Y = alpha * (T_p @ Y) + (1 - alpha) * Y0
    return Y.astype(np.float32)

for a in ALPHAS:
    LP[a] = run_lp(a)

lp_probs = np.stack([LP[a][te_idx] for a in ALPHAS], axis=0)  # (9, 2751, 10)
print(f'  LP probs: {lp_probs.shape}', flush=True)

F_IN = F_BASE + NC + 3  # 780

# ═══════════════════════════════════════════════
# PART 2: SAGE训练 (200 voters)
# ═══════════════════════════════════════════════
print('\n[2/6] Training SAGE (200 voters)...', flush=True)
EPOCHS = 600
LR_SAGE = 0.005
WD = 5e-4
DROP = 0.5
lock = threading.Lock()

def train_sage_config(cfg):
    """训练一个SAGE配置的所有seed"""
    arch, hdim, nlayers, heads, alpha, nseeds = cfg
    a_str = f'{alpha:.2f}'
    X_np = np.hstack([feat, LP[alpha], deg_feat]).astype(np.float32)
    X = torch.tensor(X_np, device=DEV)
    probs = []

    tag = f'{arch}-h{hdim}-L{nlayers}-h{heads}-a{a_str}'
    start = time.time()

    for s in range(nseeds):
        torch.manual_seed(s + hash(tag) % 10000)
        layers = []
        for i in range(nlayers):
            in_d = (F_IN * 2) if i == 0 else (hdim * 2)
            out_d = hdim if i < nlayers - 1 else NC
            layers.append(nn.Linear(in_d, out_d, device=DEV))
        params = [p for l in layers for p in l.parameters()]
        opt = torch.optim.AdamW(params, lr=LR_SAGE, weight_decay=WD)
        for _ in range(EPOCHS):
            for l in layers:
                l.train()
            opt.zero_grad()
            h = X
            for i in range(nlayers - 1):
                h_neigh = A_sp @ h
                h_cat = torch.cat([h, h_neigh], dim=-1)
                h = F.relu(F.dropout(layers[i](h_cat), p=DROP, training=True))
            F.cross_entropy(layers[-1](torch.cat([h, A_sp @ h], dim=-1))[tr_t], Y_t[tr_t]).backward()
            opt.step()
        for l in layers:
            l.eval()
        with torch.no_grad():
            h = X
            for i in range(nlayers - 1):
                h_neigh = A_sp @ h
                h_cat = torch.cat([h, h_neigh], dim=-1)
                h = F.relu(F.dropout(layers[i](h_cat), p=DROP, training=False))
            probs.append(F.softmax(layers[-1](torch.cat([h, A_sp @ h], dim=-1)), dim=-1).cpu().numpy()[te_idx])

    elapsed = time.time() - start
    with lock:
        print(f'  [SAGE] h{hdim} L{nlayers} H{heads} a{a_str}: {nseeds}s, {elapsed:.0f}s', flush=True)
    return np.stack(probs, axis=0)

# 20 SAGE configs × 10 seeds = 200 voters
sage_configs = [
    ('SAGE', 32, 2, 0, 0.80, 10),
    ('SAGE', 32, 2, 0, 0.90, 10),
    ('SAGE', 64, 2, 0, 0.80, 10),
    ('SAGE', 64, 2, 0, 0.85, 10),
    ('SAGE', 64, 2, 0, 0.90, 10),
    ('SAGE', 64, 2, 0, 0.95, 10),
    ('SAGE', 64, 3, 0, 0.85, 10),
    ('SAGE', 64, 3, 0, 0.90, 10),
    ('SAGE', 128, 2, 0, 0.85, 10),
    ('SAGE', 128, 2, 0, 0.90, 10),
    ('SAGE', 128, 2, 0, 0.95, 10),
    ('SAGE', 128, 3, 0, 0.85, 10),
    ('SAGE', 128, 3, 0, 0.90, 10),
    ('SAGE', 128, 3, 0, 0.93, 10),
    ('SAGE', 128, 3, 0, 0.95, 10),
    ('SAGE', 256, 2, 0, 0.90, 10),
    ('SAGE', 256, 3, 0, 0.93, 10),
    ('SAGE', 256, 3, 0, 0.95, 10),
    ('SAGE', 256, 3, 0, 0.97, 10),
    ('SAGE', 256, 4, 0, 0.97, 10),
    ('SAGE', 512, 3, 0, 0.95, 10),
    ('SAGE', 512, 3, 0, 0.97, 10),
    ('SAGE', 512, 3, 0, 0.99, 10),
    ('SAGE', 512, 4, 0, 0.99, 10),
]

t0 = time.time()
all_sage_probs = []
for i, cfg in enumerate(sage_configs):
    p = train_sage_config(cfg)
    all_sage_probs.append(p)
    elapsed = time.time() - t0
    done = i + 1
    print(f'  Progress: {done}/{len(sage_configs)} ({100*done/len(sage_configs):.0f}%), {elapsed:.0f}s elapsed', flush=True)

sage_probs = np.concatenate(all_sage_probs, axis=0)  # (200, 2751, 10)
print(f'  SAGE probs: {sage_probs.shape}, {time.time()-t0:.0f}s', flush=True)

# ═══════════════════════════════════════════════
# PART 3: Node2Vec训练 (15 voters: 3p/q × 5seeds)
# ═══════════════════════════════════════════════
print('\n[3/6] Training Node2Vec + SVM (15 voters)...', flush=True)
try:
    from torch_geometric.nn import Node2Vec
    from torch_geometric.utils import from_scipy_sparse_matrix
    from torch_geometric.data import Data

    edge_index = from_scipy_sparse_matrix(adj_raw.tocoo())[0]
    data_tg = Data(edge_index=edge_index, num_nodes=N)

    n2v_probs_all = []
    n2v_configs = [(1, 1), (1, 2), (2, 1)]
    for pi, (p, q) in enumerate(n2v_configs):
        for seed in range(5):
            tag = f'p{p}_q{q}_s{seed}'
            s0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = Node2Vec(data_tg.edge_index, embedding_dim=128, walk_length=20,
                             context_size=10, walks_per_node=10, p=p, q=q, sparse=True).to(DEV)
            optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                loss = model.loss()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                emb = model.embedding.weight.cpu().numpy()

            clf = SVC(probability=True, kernel='rbf', random_state=seed)
            clf.fit(emb[tr_idx], labels[tr_idx])
            probs = clf.predict_proba(emb[te_idx])
            n2v_probs_all.append(probs)
            print(f'  Node2Vec [{tag}]: {time.time()-s0:.0f}s', flush=True)

    n2v_probs = np.stack(n2v_probs_all, axis=0)  # (15, 2751, 10)
    print(f'  Node2Vec probs: {n2v_probs.shape}', flush=True)
except ImportError as e:
    print(f'  ⚠ torch_geometric not available ({e}), using placeholder', flush=True)
    # 如果torch_geometric不可用，用随机嵌入替代（不影响结果）
    n2v_probs = np.zeros((0, N_TEST, NC))

# ═══════════════════════════════════════════════
# PART 4: 特征工程 (LogReg on degree + PageRank)
# ═══════════════════════════════════════════════
print('\n[4/6] Feature engineering (deg + PageRank)...', flush=True)

# PageRank 幂迭代
s0 = time.time()
pr = np.ones(N, dtype=np.float64) / N
for k in range(100):
    pr_new = 0.85 * (T_p.T @ pr) + 0.15 / N
    if np.abs(pr_new - pr).sum() < 1e-8:
        break
    pr = pr_new
print(f'  PageRank: {k+1} iters, {time.time()-s0:.0f}s', flush=True)

# 特征组合
log_deg = np.log1p(deg).astype(np.float64)
feat_eng = np.stack([log_deg, pr], axis=1)  # (N, 2)
feat_eng = StandardScaler().fit_transform(feat_eng.astype(np.float64))

# 用训练集标签训练Logistic Regression
clf_lr = LogisticRegression(max_iter=1000, multi_class='multinomial', C=1.0, random_state=SEED)
clf_lr.fit(feat_eng[tr_idx], labels[tr_idx])
feat_probs = clf_lr.predict_proba(feat_eng[te_idx])  # (2751, 10)
feat_pred = feat_probs.argmax(1)
print(f'  Feat pred dist: {dict(sorted(Counter(feat_pred).items()))}', flush=True)
print(f'  Feat probs: {feat_probs.shape}', flush=True)

# ═══════════════════════════════════════════════
# PART 5: 加权集成投票
# ═══════════════════════════════════════════════
print('\n[5/6] Weighted ensemble voting...', flush=True)

# 权重策略:
# - SAGE: weight=3 (最可靠，200 voters)
# - LP: weight=1 (9 voters)
# - Node2Vec: weight=2 (15 voters, 捕捉结构)
# - Feat: weight=1 (1 voter, 简单baseline)

all_probs_list = []
all_weights_list = []

# SAGE (weight=3)
sage_weight = 3.0
all_probs_list.append(sage_probs)
all_weights_list.append(np.full(sage_probs.shape[0], sage_weight))

# LP (weight=1)
lp_weight = 1.0
all_probs_list.append(lp_probs)
all_weights_list.append(np.full(lp_probs.shape[0], lp_weight))

# Node2Vec (weight=2)
if n2v_probs.shape[0] > 0:
    n2v_weight = 2.0
    all_probs_list.append(n2v_probs)
    all_weights_list.append(np.full(n2v_probs.shape[0], n2v_weight))

# Feat (weight=1)
feat_weight = 1.0
all_probs_list.append(feat_probs[None, :, :])
all_weights_list.append(np.array([feat_weight]))

# 合并
all_probs = np.concatenate(all_probs_list, axis=0)
all_weights = np.concatenate(all_weights_list, axis=0)

print(f'  Total voters: {len(all_probs)} (weighted)', flush=True)
print(f'  Weights: SAGE=3, LP=1, Node2Vec=2, Feat=1', flush=True)

# 加权平均
weighted_prob = np.average(all_probs, axis=0, weights=all_weights)
final = weighted_prob.argmax(axis=1)
dist = Counter(final)
print(f'  Prediction dist: {dict(sorted(dist.items()))}', flush=True)

# ═══════════════════════════════════════════════
# PART 6: 保存
# ═══════════════════════════════════════════════
print('\n[6/6] Saving results...', flush=True)

import pandas as pd
out_csv = os.path.join(OUT, 'A1.csv')
df = pd.DataFrame({'test_idx': te_idx, 'label': final})
df.to_csv(out_csv, index=False)

# 方法间分歧分析
sage_avg = sage_probs.mean(0).argmax(1)
lp_avg = lp_probs.mean(0).argmax(1)
feat_avg = feat_probs.argmax(1)
print(f'  SAGE vs LP disagreement: {(sage_avg != lp_avg).sum()}/{N_TEST} ({100*(sage_avg != lp_avg).sum()/N_TEST:.1f}%)', flush=True)
print(f'  SAGE vs Feat disagreement: {(sage_avg != feat_avg).sum()}/{N_TEST} ({100*(sage_avg != feat_avg).sum()/N_TEST:.1f}%)', flush=True)

total_time = time.time() - t0
print(f'\n{"="*60}', flush=True)
print(f'SAVED to {out_csv}', flush=True)
print(f'Methods: SAGE({sage_probs.shape[0]}) + LP({lp_probs.shape[0]}) + Node2Vec({n2v_probs.shape[0]}) + Feat(1)', flush=True)
print(f'Total voters: {all_probs.shape[0]} (weighted)', flush=True)
print(f'Time: {total_time:.0f}s ({total_time/60:.1f}min)', flush=True)
print(f'{"="*60}', flush=True)
print(f'\n[FINAL RESULT] Voters: {all_probs.shape[0]} | '
      f'dist: {dict(sorted(dist.items()))} | '
      f'{total_time/60:.1f}min', flush=True)
