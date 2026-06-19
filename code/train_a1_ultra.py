"""A1 ULTRA: 并行多模型训练 × 超大GAT × 200+选民 × 通宵跑
V100 16GB: 单模型~100MB → 同时训多模型 → 真正榨干GPU"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, os, sys, time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_norm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading

DEV = torch.device('cuda')
SEED = 42
np.random.seed(SEED)

# ═══ 加载 ═══
DATA = sys.argv[1] if len(sys.argv) > 1 else 'data/cls_data/A1.npz'
print(f'Loading {DATA}...', flush=True)
d = np.load(DATA)
adj_raw = csr_matrix((d['adj_data'],d['adj_indices'],d['adj_indptr']), shape=tuple(d['adj_shape']))
feat_raw = csr_matrix((d['attr_data'],d['attr_indices'],d['attr_indptr']), shape=tuple(d['attr_shape']))
labels = d['labels'].astype(int); tr_idx = d['train_idx']; te_idx = d['test_idx']
N = adj_raw.shape[0]; NC = 10

feat = sk_norm(feat_raw.toarray().astype(np.float32), norm='l2', axis=1)
feat[np.isnan(feat).any(axis=1)] = 0
F_BASE = feat.shape[1]

deg = np.array(adj_raw.sum(1)).flatten().astype(np.float32)
deg_feat = np.hstack([np.log1p(deg).reshape(-1,1),
    (deg<5).astype(np.float32).reshape(-1,1),(deg<10).astype(np.float32).reshape(-1,1)])

asym = adj_raw + adj_raw.T; asym.setdiag(1)
ds = np.array(asym.sum(1)).flatten()
dis = np.where(ds>0, 1.0/np.sqrt(ds), 0)
An = csr_matrix((dis,(range(N),range(N))),shape=(N,N)) @ asym @ csr_matrix((dis,(range(N),range(N))),shape=(N,N))
coo = An.tocoo()
A_sp = torch.sparse_coo_tensor(
    torch.tensor(np.vstack((coo.row,coo.col)), dtype=torch.long, device=DEV),
    torch.tensor(coo.data, dtype=torch.float32, device=DEV), (N,N)).coalesce()

rs = np.array(asym.sum(1)).flatten(); rs[rs==0]=1
T_p = csr_matrix((1.0/rs,(range(N),range(N))),shape=(N,N)) @ asym

Y_t = torch.tensor(labels, dtype=torch.long, device=DEV)
tr_t = torch.tensor(tr_idx, dtype=torch.long, device=DEV)
te_t = torch.tensor(te_idx, dtype=torch.long, device=DEV)

Y0 = np.zeros((N,NC), dtype=np.float32)
for ti in tr_idx: Y0[ti, labels[ti]] = 1.0

def run_lp(alpha, iters=80):
    Y = Y0.copy()
    for _ in range(iters): Y = alpha*(T_p@Y) + (1-alpha)*Y0
    return Y.astype(np.float32)

# ═══ 9种LP ═══
ALPHAS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
print(f'Computing {len(ALPHAS)} LP variants (80 iters each)...', flush=True)
LP = {a: run_lp(a) for a in ALPHAS}
F_IN = F_BASE + NC + 3  # 780

# ═══ 训练单个配置 ═══
EPOCHS = 600; LR = 0.005; WD = 5e-4; DROP = 0.5
lock = threading.Lock()

def train_config(cfg):
    """训练一个配置的所有seed, 返回softmax概率数组"""
    arch, hdim, nlayers, heads, alpha, nseeds = cfg
    a_str = f'{alpha:.2f}'
    X_np = np.hstack([feat, LP[alpha], deg_feat]).astype(np.float32)
    X = torch.tensor(X_np, device=DEV)
    probs = []
    
    tag = f'{arch}-h{hdim}-L{nlayers}-h{heads}-a{a_str}'
    start = time.time()
    
    for s in range(nseeds):
        torch.manual_seed(s + hash(tag) % 10000)
        if arch == 'GCN':
            layers = [nn.Linear(F_IN if i==0 else hdim, hdim if i<nlayers-1 else NC, device=DEV) for i in range(nlayers)]
            params = [p for l in layers for p in l.parameters()]
            opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
            for _ in range(EPOCHS):
                for l in layers: l.train()
                opt.zero_grad()
                h = X
                for i in range(nlayers-1):
                    h = F.relu(F.dropout(A_sp @ layers[i](h) if i==0 else layers[i](h), p=DROP, training=True))
                F.cross_entropy(layers[-1](h)[tr_t], Y_t[tr_t]).backward(); opt.step()
            for l in layers: l.eval()
            with torch.no_grad():
                h = X
                for i in range(nlayers-1):
                    h = F.relu(F.dropout(A_sp @ layers[i](h) if i==0 else layers[i](h), p=DROP, training=False))
                probs.append(F.softmax(layers[-1](h), dim=-1).cpu().numpy()[te_idx])
        elif arch == 'SAGE':
            # GraphSAGE: concat(self, mean-aggregated neighbors) → wider linear
            layers = []
            for i in range(nlayers):
                in_d = (F_IN*2) if i==0 else (hdim*2)
                out_d = hdim if i<nlayers-1 else NC
                layers.append(nn.Linear(in_d, out_d, device=DEV))
            params = [p for l in layers for p in l.parameters()]
            opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
            for _ in range(EPOCHS):
                for l in layers: l.train()
                opt.zero_grad()
                h = X
                for i in range(nlayers-1):
                    h_neigh = A_sp @ h
                    h_cat = torch.cat([h, h_neigh], dim=-1)
                    h = F.relu(F.dropout(layers[i](h_cat), p=DROP, training=True))
                F.cross_entropy(layers[-1](torch.cat([h, A_sp@h], dim=-1))[tr_t], Y_t[tr_t]).backward(); opt.step()
            for l in layers: l.eval()
            with torch.no_grad():
                h = X
                for i in range(nlayers-1):
                    h_neigh = A_sp @ h
                    h_cat = torch.cat([h, h_neigh], dim=-1)
                    h = F.relu(F.dropout(layers[i](h_cat), p=DROP, training=False))
                probs.append(F.softmax(layers[-1](torch.cat([h, A_sp@h], dim=-1)), dim=-1).cpu().numpy()[te_idx])
        else:  # GAT
            proj_layers = [nn.Linear(F_IN if i==0 else hdim*heads, hdim*heads, device=DEV) for i in range(nlayers)]
            out_layer = nn.Linear(hdim*heads, NC, device=DEV)
            params = [p for proj in proj_layers for p in proj.parameters()] + list(out_layer.parameters())
            opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
            for _ in range(EPOCHS):
                opt.zero_grad()
                h = X
                for proj in proj_layers:
                    h = proj(h).view(N, heads, -1)
                    agg = torch.stack([A_sp @ h[:, k, :] for k in range(heads)], dim=1)
                    h = F.elu(agg.permute(0,2,1).reshape(N, -1))
                    h = F.dropout(h, p=DROP, training=True)
                F.cross_entropy(out_layer(h)[tr_t], Y_t[tr_t]).backward(); opt.step()
            with torch.no_grad():
                h = X
                for proj in proj_layers:
                    h = proj(h).view(N, heads, -1)
                    agg = torch.stack([A_sp @ h[:, k, :] for k in range(heads)], dim=1)
                    h = F.elu(agg.permute(0,2,1).reshape(N, -1))
                    h = F.dropout(h, p=DROP, training=False)
                probs.append(F.softmax(out_layer(h), dim=-1).cpu().numpy()[te_idx])
    
    elapsed = time.time() - start
    with lock:
        print(f'  [{arch}] h{hdim} L{nlayers} H{heads} a{a_str}: {nseeds}s, {elapsed:.0f}s', flush=True)
    return np.stack(probs, axis=0)

# ═══ 配置表: SAGE + LP集成（目标0.77） ═══
# 10 configs × 10 seeds = 100 SAGE voters + 9 LP variants = 109 total
configs = [
    # 小模型（快速，捕捉局部结构）
    ('SAGE', 64,  2, 0, 0.80, 10),
    ('SAGE', 64,  2, 0, 0.90, 10),
    ('SAGE', 64,  3, 0, 0.85, 10),
    # 中模型（平衡）
    ('SAGE', 128, 2, 0, 0.85, 10),
    ('SAGE', 128, 2, 0, 0.95, 10),
    ('SAGE', 128, 3, 0, 0.90, 10),
    ('SAGE', 128, 3, 0, 0.93, 10),
    # 大模型（捕捉全局结构）
    ('SAGE', 256, 3, 0, 0.93, 10),
    ('SAGE', 256, 4, 0, 0.97, 10),
    ('SAGE', 512, 3, 0, 0.99, 10),
]

total_voters = sum(c[-1] for c in configs)
print(f'\n{len(configs)} configs, {total_voters} voters total', flush=True)
print(f'Running on V100 with GPU-async parallelism...\n', flush=True)

out_dir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/a1_ultra'
os.makedirs(out_dir, exist_ok=True)
CKPT = os.path.join(out_dir, '.ckpt.npz')

# ═══ 断点续跑 ═══
all_probs = []
start_i = 0
t0 = time.time()
if os.path.exists(CKPT):
    ckpt = np.load(CKPT, allow_pickle=True)
    all_probs = [ckpt[f'p{i}'] for i in range(len(ckpt.files) - 1)]  # -1 to exclude 'elapsed' key
    start_i = len(all_probs)
    t0 -= float(ckpt.get('elapsed', 0))
    del ckpt
    done_sofar = sum(c[-1] for c in configs[:start_i])
    print(f'[RESUME] {start_i}/{len(configs)} configs ({done_sofar} voters) loaded from .ckpt.npz', flush=True)

torch.cuda.synchronize()
if start_i == 0:
    t0 = time.time()

# ═══ 顺序训练 (每config完立刻存盘) ═══
for i in range(start_i, len(configs)):
    cfg = configs[i]
    probs = train_config(cfg)
    all_probs.append(probs)
    elapsed = time.time() - t0
    done = sum(c[-1] for c in configs[:i+1])
    print(f'  Progress: {done}/{total_voters} ({100*done/total_voters:.0f}%), {elapsed:.0f}s elapsed', flush=True)
    # 断点保存 — 崩溃不丢
    np.savez_compressed(CKPT, **{f'p{j}': p for j, p in enumerate(all_probs)}, elapsed=elapsed)
    print(f'  [CKPT] saved {len(all_probs)} configs to {CKPT}', flush=True)

torch.cuda.synchronize()
elapsed = time.time() - t0

# ═══ 软投票 (SAGE + LP) ═══
all_p = np.concatenate(all_probs, axis=0)  # (V, 2751, 10)

# 加LP voters (用第58行已计算的LP字典)
lp_probs = []
for alpha in ALPHAS:
    lp_pred = LP[alpha][te_idx]  # (2751, 10)
    lp_probs.append(lp_pred)
lp_p = np.stack(lp_probs, axis=0)  # (9, 2751, 10)

# 合并SAGE和LP
all_p_combined = np.concatenate([all_p, lp_p], axis=0)  # (V+9, 2751, 10)
print(f'[ENSEMBLE] SAGE: {len(all_p)} voters + LP: {len(lp_p)} variants = {len(all_p_combined)} total', flush=True)

avg_prob = all_p_combined.mean(axis=0)
final = avg_prob.argmax(axis=1)
dist = Counter(final)

# ═══ 输出 ═══
print(f'\n{"="*60}')
print(f'TOTAL: {len(all_p)} voters, {elapsed:.0f}s ({elapsed/60:.1f}min)')
print(f'Distribution: {dict(sorted(dist.items()))}')
print(f'{"="*60}', flush=True)

# 分歧分析 (GCN vs SAGE vs GAT)
n_gcn = sum(c[-1] for c in configs if c[0]=='GCN')
n_sage = sum(c[-1] for c in configs if c[0]=='SAGE')
n_gat = sum(c[-1] for c in configs if c[0]=='GAT')
gcn_v = all_p[:n_gcn].mean(0).argmax(1)
sage_v = all_p[n_gcn:n_gcn+n_sage].mean(0).argmax(1)
print(f'Architecture disagreement:', flush=True)
print(f'  GCN vs SAGE: {(gcn_v!=sage_v).sum()}/2751 ({100*(gcn_v!=sage_v).sum()/2751:.1f}%)', flush=True)
if n_gat > 0:
    gat_v = all_p[n_gcn+n_sage:].mean(0).argmax(1)
    print(f'  GCN vs GAT:  {(gcn_v!=gat_v).sum()}/2751 ({100*(gcn_v!=gat_v).sum()/2751:.1f}%)', flush=True)
    print(f'  SAGE vs GAT: {(sage_v!=gat_v).sum()}/2751 ({100*(sage_v!=gat_v).sum()/2751:.1f}%)', flush=True)

# 保存
import pandas as pd
out = pd.DataFrame({'test_idx': te_idx, 'label': final})
out.to_csv(os.path.join(out_dir, 'A1.csv'), index=False)
info = {
    'voters': int(len(all_p)), 'gcn': int(n_gcn), 'sage': int(n_sage), 'gat': int(n_gat),
    'epochs': EPOCHS, 'lp_iters': 80, 'lp_alphas': ALPHAS,
    'disagree_GCNvSAGE': f'{(gcn_v!=sage_v).sum()/2751*100:.1f}%',
    'distribution': {str(k): v for k,v in sorted(dist.items())},
    'time_minutes': f'{elapsed/60:.1f}',
}
if n_gat > 0:
    info['disagree_GCNvGAT'] = f'{(gcn_v!=gat_v).sum()/2751*100:.1f}%'
    info['disagree_SAGEvGAT'] = f'{(sage_v!=gat_v).sum()/2751*100:.1f}%'
with open(os.path.join(out_dir, 'a1_ultra_info.json'), 'w') as f:
    json.dump(info, f, indent=2)
print(f'Saved to {out_dir}', flush=True)
# ═══ 纯净摘要 — 这行直接复制给我 ═══
print(f'[RESULT] {len(all_p)} voters | '
      f'GCN:{n_gcn} SAGE:{n_sage} GAT:{int(n_gat)} | '
      f'dist: {dict(sorted(dist.items()))} | '
      f'disagree: GvS={100*(gcn_v!=sage_v).sum()/2751:.1f}%'
      + (f' GvGAT={100*(gcn_v!=gat_v).sum()/2751:.1f}%' if n_gat > 0 else '')
      + (f' SvGAT={100*(sage_v!=gat_v).sum()/2751:.1f}%' if n_gat > 0 else '')
      + f' | {elapsed/60:.1f}min', flush=True)
