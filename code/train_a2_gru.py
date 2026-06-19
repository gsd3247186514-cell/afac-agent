"""A2 GRU4Rec + ItemCF Reranking — 零折扣训练
修复V22三大漏洞:
1. 用户级split（非样例级）
2. 短前缀验证（模拟测试分布）
3. 验证时也用ItemCF重排序
目标: val NDCG → test NDCG 不打折
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter, defaultdict
import time, os, random, math, sys

DATA = r'C:\Users\高帅东\Desktop\tianchi_afac\A_recommend'
DEV = torch.device('cuda')
MAX_LEN = 50
EMB_DIM = 128; HIDDEN = 128; BATCH = 256; EPOCHS = 30; PATIENCE = 8
N_NEG = 5; LR = 1e-3; WD = 1e-4
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ═══ Load ═══
print("[GRU] Loading...", flush=True)
train_df = pd.read_csv(os.path.join(DATA, 'train.csv'))
test_df  = pd.read_csv(os.path.join(DATA, 'test.csv'))
item_df  = pd.read_csv(os.path.join(DATA, 'item.csv'))
user_df  = pd.read_csv(os.path.join(DATA, 'user.csv'))

all_iids = item_df['iid'].tolist(); NI = len(all_iids)
iid2idx = {iid: i for i,iid in enumerate(all_iids)}
idx2iid = {i: iid for i,iid in enumerate(all_iids)}

# User features
uf_cols = [c for c in user_df.columns if c != 'uid']
for c in uf_cols:
    if user_df[c].dtype == 'object': user_df[c]=user_df[c].astype('category').cat.codes
    user_df[c]=user_df[c].fillna(0).astype(np.float32)
NF_USER = len(uf_cols)
uid2idx = {uid: i for i,uid in enumerate(user_df['uid'])}
uf_raw = user_df[uf_cols].values
uf_m = uf_raw.mean(0,keepdims=True); uf_s = uf_raw.std(0,keepdims=True).clip(min=1e-8)
user_feat_n = (uf_raw-uf_m)/uf_s

def pp(s):
    if pd.isna(s) or str(s) in ('nan',''): return []
    return [x.strip() for x in str(s).split(',') if x.strip()]

# ═══ User-level sequences ═══
print("[GRU] Building user sequences...", flush=True)
user_items = {}; user_targets = {}; user_feat_idx = {}
for i in range(len(train_df)):
    row = train_df.iloc[i]
    uid = row['uid']
    raw = pp(row['item_seq_raw'])
    items = [iid2idx[iid] for iid in raw[-MAX_LEN:] if iid in iid2idx]
    t_idx = iid2idx.get(row['target_iid'])
    u_idx = uid2idx.get(uid)
    if not items or t_idx is None or u_idx is None: continue
    user_items[uid] = items
    user_targets[uid] = t_idx
    user_feat_idx[uid] = u_idx

all_users = sorted(user_items.keys())
n_val = len(all_users) // 5
random.shuffle(all_users)
val_users = set(all_users[:n_val])
train_users = [u for u in all_users if u not in val_users]
print(f"  Train users: {len(train_users)}, Val users: {n_val}", flush=True)

# Build training examples from train users
train_exs = []
for uid in train_users:
    items = user_items[uid]
    for i in range(1, len(items)):
        train_exs.append((uid, items[:i], items[i]))
print(f"  Train examples: {len(train_exs)}", flush=True)

# Val examples: SHORT PREFIX only (mimics test: median 3 items)
val_exs = []
for uid in val_users:
    items = user_items[uid]
    for k in range(1, min(len(items), 6)):  # max 5-item prefix
        val_exs.append((uid, items[:k], items[k]))
print(f"  Val examples (short prefix): {len(val_exs)}", flush=True)

# ═══ ItemCF ═══
print("[GRU] Building ItemCF...", flush=True)
item_targets = defaultdict(Counter); item_pop_count = Counter()
for i in range(len(train_df)):
    row = train_df.iloc[i]
    dedup = pp(row['item_seq_dedup']); target = row['target_iid']
    for p, iid in enumerate(dedup):
        pos_w = np.exp(0.3*(p-len(dedup)+1))
        item_targets[iid][target] += pos_w; item_pop_count[iid] += 1
for iid in item_targets:
    idf = np.log(40000/(1+item_pop_count[iid]))
    for t in item_targets[iid]: item_targets[iid][t] *= idf

pop = np.zeros(NI, dtype=np.float32)
for iid,c in Counter(train_df['target_iid']).items():
    idx = iid2idx.get(iid)
    if idx is not None: pop[idx] = c
pop /= pop.sum()

hot_cnt = Counter(train_df['target_iid'])
hot_pn = np.ones(NI, dtype=np.float32)
for iid,cnt in hot_cnt.items():
    idx = iid2idx.get(iid)
    if idx is not None: hot_pn[idx] = 1.0/(1.0+0.5*np.log(1+cnt))

def get_candidates(dedup_items, n_cand=200):
    scores = np.zeros(NI, dtype=np.float32)
    n = len(dedup_items)
    n_c = 300 if n<=2 else 200 if n<=5 else 150
    iw = 2.0 if n<=2 else 3.0 if n<=5 else 5.0
    for p,iid in enumerate(dedup_items):
        if iid not in item_targets: continue
        tc = item_targets[iid]; tot = sum(tc.values())
        if tot == 0: continue
        rec = np.exp(0.3*(p-n+1))
        for t,c in tc.most_common(n_c):
            idx = iid2idx.get(t)
            if idx is not None: scores[idx] += iw*rec*c/tot
    hb = 0.3 if n<=2 else 0.15 if n<=5 else 0.05
    for iid in dedup_items:
        idx = iid2idx.get(iid)
        if idx is not None: scores[idx] += hb
    scores *= hot_pn; scores += 0.15*pop
    mx = scores.max()
    if mx > 0: scores /= mx
    top = np.argsort(-scores)[:n_cand]
    return list(top), scores[top]

# ═══ GRU Model ═══
print("[GRU] Training...", flush=True)
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NI, EMB_DIM)
        self.gru = nn.GRU(EMB_DIM, HIDDEN, num_layers=2, batch_first=True, dropout=0.2)
        self.user_proj = nn.Linear(NF_USER, HIDDEN) if NF_USER>0 else None
        self.out = nn.Linear(HIDDEN, EMB_DIM)
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)
    def forward(self, seqs, uf=None):
        emb = self.emb(seqs)
        lengths = (seqs>0).sum(1)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        urep = h_n[-1]
        if self.user_proj is not None and uf is not None:
            urep = urep+self.user_proj(uf)
        return self.out(urep)

model = GRUModel().to(DEV)
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

# Neg sampling weights
tgt_cnt = Counter(user_targets.values())
pw = np.array([tgt_cnt.get(i,1) for i in range(NI)], dtype=np.float32)
pw = np.power(pw, 0.75); pw /= pw.sum()

# User feature cache
uf_cache = {}
for uid, idx in user_feat_idx.items():
    uf_cache[uid] = torch.tensor(user_feat_n[idx], dtype=torch.float32)

best_val = 0; best_state = None; pc = 0
for ep in range(EPOCHS):
    model.train(); tl,nb = 0,0
    random.shuffle(train_exs)
    for bs in range(0, len(train_exs), BATCH):
        batch = train_exs[bs:bs+BATCH]; bsn = len(batch)
        ml = min(max(len(s) for _,s,_ in batch), MAX_LEN)
        pad = torch.zeros(bsn, ml, dtype=torch.long, device=DEV)
        uf = torch.zeros(bsn, NF_USER, device=DEV)
        tgt_t = torch.zeros(bsn, dtype=torch.long, device=DEV)
        for i,(uid,s,t) in enumerate(batch):
            pad[i,:len(s)] = torch.tensor(s[:ml], dtype=torch.long)
            uf[i] = uf_cache[uid]
            tgt_t[i] = t
        urep = model(pad, uf)
        scores = urep @ model.emb.weight.T
        
        pos_s = scores[torch.arange(bsn), tgt_t]
        neg = np.random.choice(NI, size=(bsn,N_NEG), p=pw)
        neg_t = torch.tensor(neg, dtype=torch.long, device=DEV)
        neg_s = scores[torch.arange(bsn).unsqueeze(1), neg_t]
        all_s = torch.cat([pos_s.unsqueeze(1), neg_s], dim=1)
        loss = F.cross_entropy(all_s, torch.zeros(bsn, dtype=torch.long, device=DEV))
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tl+=loss.item(); nb+=1
    
    # Validation: with ItemCF reranking (exact inference pipeline)
    if ep%3==0 or ep>=EPOCHS-5:
        model.eval()
        # Weight search on val subset
        v_sample = random.sample(val_exs, min(2000, len(val_exs)))
        best_w = (0.3, 0.5); best_v = 0
        
        for wg in [0.2, 0.3, 0.4]:
            for wi in [0.3, 0.4, 0.5, 0.6]:
                if wg+wi > 0.9: continue
                ndcgs = []
                for uid, pref, tgt in v_sample[:500]:
                    # ItemCF candidates from dedup (approximate from raw)
                    dedup_iids = [idx2iid[it] for it in pref]  # items → original iid
                    cands, icf_s = get_candidates(dedup_iids)
                    if not cands: continue
                    # GRU score
                    with torch.no_grad():
                        seq_t = torch.tensor(pref[-MAX_LEN:], dtype=torch.long, device=DEV).unsqueeze(0)
                        uf_t = uf_cache[uid].unsqueeze(0).to(DEV)
                        urep = model(seq_t, uf_t)
                        cand_t = torch.tensor(cands, dtype=torch.long, device=DEV)
                        gru_s = (urep @ model.emb(cand_t)).squeeze(0).cpu().numpy()
                    gmax = gru_s.max(); icfmax = icf_s.max()
                    if gmax > 0: gru_s /= gmax
                    if icfmax > 0: icf_s /= icfmax
                    final_s = wg*gru_s + wi*icf_s
                    top10 = [cands[j] for j in np.argsort(-final_s)[:10]]
                    if tgt in top10:
                        ndcgs.append(1.0/np.log2(top10.index(tgt)+2))
                    else:
                        ndcgs.append(0.0)
                v = np.mean(ndcgs) if ndcgs else 0
                if v > best_v: best_v = v; best_w = (wg, wi)
        
        val_ndcg = best_v
        if val_ndcg > best_val:
            best_val = val_ndcg; pc = 0
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: pc += 1
        print(f"  Ep{ep:2d}: loss={tl/max(nb,1):.4f} val_ndcg={val_ndcg:.4f} best={best_val:.4f} w={best_w} pc={pc}", flush=True)
        if pc >= PATIENCE: print(f"  Early stop@{ep}", flush=True); break

model.load_state_dict(best_state); model.eval()
print(f"\n[GRU] Best val NDCG: {best_val:.4f}", flush=True)

# ═══ Full inference ═══
print("[GRU] Inference...", flush=True)
def predict(row, w_gru, w_icf):
    raw = pp(row.get('item_seq_raw',''))
    dedup = pp(row.get('item_seq_dedup',''))
    items = [iid2idx[iid] for iid in raw[-MAX_LEN:] if iid in iid2idx]
    uid = row['uid']; u_idx = uid2idx.get(uid)
    cands, icf_s = get_candidates(dedup)
    
    if not items or u_idx is None:
        top = np.argsort(-icf_s)[:10]
        return [all_iids[cands[i]] for i in top]
    
    with torch.no_grad():
        seq_t = torch.tensor(items, dtype=torch.long, device=DEV).unsqueeze(0)
        uf_t = torch.tensor(user_feat_n[u_idx], dtype=torch.float32, device=DEV).unsqueeze(0)
        urep = model(seq_t, uf_t)
        cand_t = torch.tensor(cands, dtype=torch.long, device=DEV)
        gru_s = (urep @ model.emb(cand_t)).squeeze(0).cpu().numpy()
    
    gmax = gru_s.max(); icfmax = icf_s.max()
    if gmax>0: gru_s/=gmax
    if icfmax>0: icf_s/=icfmax
    final_s = w_gru*gru_s + w_icf*icf_s
    top = np.argsort(-final_s)[:10]
    return [all_iids[cands[i]] for i in top]

preds = [','.join(predict(row, *best_w)) for _,row in test_df.iterrows()]

OUT = os.path.join(os.path.dirname(DATA) if os.path.dirname(DATA) else '.', 'a2_gru_out')
os.makedirs(OUT, exist_ok=True)
out_df = pd.DataFrame({'uid': test_df['uid'].apply(lambda x: int(str(x).lstrip('uU'))), 'prediction': preds})
out_df.to_csv(os.path.join(OUT, 'A2.csv'), index=False)
uniq = set(); [uniq.update(p.split(',')) for p in preds]
print(f"[GRU] A2.csv: {len(out_df)} rows, {len(uniq)} products", flush=True)
print(f"[GRU] best_val={best_val:.4f} w={best_w}", flush=True)
