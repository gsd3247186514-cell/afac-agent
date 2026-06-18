"""A1极限训练: 135选民 × 多样架构 × 4种LP α × 500epoch × 软投票.
在V100上跑, 预计20-30分钟出分."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, os, sys, time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_norm
from collections import Counter

DEV = torch.device('cuda')
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ═══ 加载 ═══
DATA = sys.argv[1] if len(sys.argv) > 1 else 'data/cls_data/A1.npz'
print(f'Loading {DATA}...', flush=True)
d = np.load(DATA)
adj_raw = csr_matrix((d['adj_data'],d['adj_indices'],d['adj_indptr']), shape=tuple(d['adj_shape']))
feat_raw = csr_matrix((d['attr_data'],d['attr_indices'],d['attr_indptr']), shape=tuple(d['attr_shape']))
labels = d['labels'].astype(int); tr_idx = d['train_idx']; te_idx = d['test_idx']
N = adj_raw.shape[0]; NC = 10

# 特征
feat = sk_norm(feat_raw.toarray().astype(np.float32), norm='l2', axis=1)
feat[np.isnan(feat).any(axis=1)] = 0
F_BASE = feat.shape[1]

# 度特征
deg = np.array(adj_raw.sum(1)).flatten().astype(np.float32)
deg_feat = np.hstack([np.log1p(deg).reshape(-1,1),
    (deg<5).astype(np.float32).reshape(-1,1),(deg<10).astype(np.float32).reshape(-1,1)])

# 邻接矩阵
asym = adj_raw + adj_raw.T; asym.setdiag(1)
ds = np.array(asym.sum(1)).flatten()
dis = np.where(ds>0, 1.0/np.sqrt(ds), 0)
An = csr_matrix((dis,(range(N),range(N))),shape=(N,N)) @ asym @ csr_matrix((dis,(range(N),range(N))),shape=(N,N))
coo = An.tocoo()
A_sp = torch.sparse_coo_tensor(
    torch.tensor(np.vstack((coo.row,coo.col)), dtype=torch.long, device=DEV),
    torch.tensor(coo.data, dtype=torch.float32, device=DEV), (N,N)).coalesce()

# LP传播矩阵
rs = np.array(asym.sum(1)).flatten(); rs[rs==0]=1
T_p = csr_matrix((1.0/rs,(range(N),range(N))),shape=(N,N)) @ asym

# 标签张量
Y_t = torch.tensor(labels, dtype=torch.long, device=DEV)
tr_t = torch.tensor(tr_idx, dtype=torch.long, device=DEV)
te_t = torch.tensor(te_idx, dtype=torch.long, device=DEV)

# LP初始值
Y0 = np.zeros((N,NC), dtype=np.float32)
for ti in tr_idx: Y0[ti, labels[ti]] = 1.0

def run_lp(alpha, iters=50):
    Y = Y0.copy()
    for _ in range(iters): Y = alpha*(T_p@Y) + (1-alpha)*Y0
    return Y.astype(np.float32)

# 预计算4种LP
print('Computing LP features (50 iters)...', flush=True)
LP = {a: run_lp(a) for a in [0.8, 0.9, 0.95, 0.99]}
F_IN = F_BASE + NC + 3  # 780

# 训练参数
EPOCHS = 500; LR = 0.005; WD = 5e-4; DROP = 0.6; DROP_GAT = 0.5

all_probs = []  # 存储softmax概率而非硬标签
t0 = time.time()

# ═══ 训练辅助 ═══
def train_GCN(X, hdim, nlayers, nseeds, tag):
    start = time.time()
    for s in range(nseeds):
        torch.manual_seed(s)
        layers = [nn.Linear(F_IN if i==0 else hdim, hdim if i<nlayers-1 else NC, device=DEV)
                  for i in range(nlayers)]
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
            all_probs.append(F.softmax(layers[-1](h), dim=-1).cpu().numpy()[te_idx])
    print(f'  {tag}: {nseeds} seeds, {time.time()-start:.0f}s', flush=True)

def train_GAT(X, hdim, nlayers, heads, nseeds, tag):
    start = time.time()
    for s in range(nseeds):
        torch.manual_seed(s + 500)
        proj_layers, out_layer = [], None
        for i in range(nlayers):
            in_d = F_IN if i==0 else hdim*heads
            proj = nn.Linear(in_d, hdim*heads, device=DEV)
            proj_layers.append(proj)
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
                h = F.dropout(h, p=DROP_GAT, training=True)
            F.cross_entropy(out_layer(h)[tr_t], Y_t[tr_t]).backward(); opt.step()
        with torch.no_grad():
            h = X
            for proj in proj_layers:
                h = proj(h).view(N, heads, -1)
                agg = torch.stack([A_sp @ h[:, k, :] for k in range(heads)], dim=1)
                h = F.elu(agg.permute(0,2,1).reshape(N, -1))
                h = F.dropout(h, p=DROP_GAT, training=False)
            all_probs.append(F.softmax(out_layer(h), dim=-1).cpu().numpy()[te_idx])
    print(f'  {tag}: {nseeds} seeds, {time.time()-start:.0f}s', flush=True)

# ═══ 135选民 ═══
print(f'\nTraining 135 voters (GCN + GAT) on V100...', flush=True)

# Config 1: GCN h128 L2 α=0.8 (25 seeds)
XA = torch.tensor(np.hstack([feat, LP[0.8], deg_feat]).astype(np.float32), device=DEV)
train_GCN(XA, 128, 2, 25, 'GCN-128-L2-α0.8')

# Config 2: GCN h256 L3 α=0.9 (25 seeds)
XB = torch.tensor(np.hstack([feat, LP[0.9], deg_feat]).astype(np.float32), device=DEV)
train_GCN(XB, 256, 3, 25, 'GCN-256-L3-α0.9')

# Config 3: GCN h256 L4 α=0.95 (25 seeds)
XC = torch.tensor(np.hstack([feat, LP[0.95], deg_feat]).astype(np.float32), device=DEV)
train_GCN(XC, 256, 4, 25, 'GCN-256-L4-α0.95')

# Config 4: GAT h128 L2 4head α=0.8 (20 seeds)
train_GAT(XA, 128, 2, 4, 20, 'GAT-128-L2-4h-α0.8')

# Config 5: GAT h256 L3 8head α=0.95 (20 seeds)
train_GAT(XC, 256, 3, 8, 20, 'GAT-256-L3-8h-α0.95')

# Config 6: GAT h256 L4 8head α=0.99 (20 seeds)
XD = torch.tensor(np.hstack([feat, LP[0.99], deg_feat]).astype(np.float32), device=DEV)
train_GAT(XD, 256, 4, 8, 20, 'GAT-256-L4-8h-α0.99')

elapsed = time.time() - t0
print(f'\nTraining done: {elapsed:.0f}s ({elapsed/60:.1f}min), {len(all_probs)} voters', flush=True)

# ═══ 软投票 ═══
probs = np.stack(all_probs, axis=0)  # (V, 2751, 10)
avg_prob = probs.mean(axis=0)  # (2751, 10)
final = avg_prob.argmax(axis=1)

dist = Counter(final)
print(f'Soft-vote distribution: {dict(sorted(dist.items()))}', flush=True)

# 分歧分析
gcn_mask = np.zeros(len(all_probs), dtype=bool)
gcn_mask[:75] = True  # first 75 are GCN
gat_mask = ~gcn_mask
gcn_vote = probs[gcn_mask].mean(0).argmax(1)
gat_vote = probs[gat_mask].mean(0).argmax(1)
disagree = (gcn_vote != gat_vote).sum()
print(f'GCN vs GAT disagree: {disagree}/2751 ({100*disagree/2751:.1f}%)', flush=True)

# ═══ 保存 ═══
import pandas as pd
out_dir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/a1_out'
os.makedirs(out_dir, exist_ok=True)
out = pd.DataFrame({'test_idx': te_idx, 'label': final})
out.to_csv(os.path.join(out_dir, 'A1.csv'), index=False)
print(f'Saved A1.csv to {out_dir}', flush=True)

# 签名
info = {
    'voters': len(all_probs), 'gcn': 75, 'gat': 60,
    'epochs': EPOCHS, 'lp_iters': 50, 'vote': 'soft',
    'gcn_gat_disagree_pct': f'{100*disagree/2751:.1f}%',
    'distribution': dict(sorted(dist.items())),
    'time_seconds': elapsed,
}
with open(os.path.join(out_dir, 'a1_max_info.json'), 'w') as f:
    json.dump(info, f, indent=2)
print(f'Info saved. Done.', flush=True)
