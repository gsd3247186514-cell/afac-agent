"""A2 V31: SASRec with max_len=50 + pop-aware neg + Dropout 0.3, 5 seeds"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
import time, os, random

DATA = r'C:\Users\高帅东\Desktop\tianchi_afac\A_recommend'
OUT = r'C:\Users\高帅东\Desktop\tianchi_afac\v31_models'
os.makedirs(OUT, exist_ok=True)
DEV = torch.device('cuda')
print(f'Device: {DEV}')

# ── Config ──
MAX_LEN = 50        # ← 匹配测试用户中位数3
EMB_DIM = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.3       # ← 更强的正则化防过拟合
BATCH = 256
EPOCHS = 80
NEG = 10            # ← 更多负样本
SEEDS = [42, 123, 456, 789, 1011]

# ── Load ──
print('Loading data...')
train_df = pd.read_csv(os.path.join(DATA, 'train.csv'))
item_df = pd.read_csv(os.path.join(DATA, 'item.csv'))
user_df = pd.read_csv(os.path.join(DATA, 'user.csv'))
all_iids = item_df['iid'].tolist(); NI = len(all_iids)
iid2idx = {iid: i for i, iid in enumerate(all_iids)}

uf_cols = [c for c in user_df.columns if c != 'uid']
for c in uf_cols:
    if user_df[c].dtype == 'object': user_df[c] = user_df[c].astype('category').cat.codes
    user_df[c] = user_df[c].fillna(0).astype(np.float32)
NF_USER = len(uf_cols)
uid2idx = {uid: i for i, uid in enumerate(user_df['uid'])}
uf_raw = user_df[uf_cols].values
uf_m = uf_raw.mean(0, keepdims=True); uf_s = uf_raw.std(0, keepdims=True).clip(min=1e-8)
user_feat_n = (uf_raw - uf_m) / uf_s

def pp(s):
    if pd.isna(s) or str(s) in ('nan', ''): return []
    return [x.strip() for x in str(s).split(',') if x.strip()]

# Build sequences
print('Building sequences...')
user_sequences = []
for i in range(len(train_df)):
    row = train_df.iloc[i]
    raw = pp(row['item_seq_raw'])
    items = [iid2idx[iid] for iid in raw if iid in iid2idx]
    u_idx = uid2idx.get(row['uid'])
    if len(items) >= 2 and u_idx is not None:
        user_sequences.append((items[-MAX_LEN:], u_idx))  # truncate early!
N_USERS = len(user_sequences)
print(f'  Users: {N_USERS}')

# Pop-aware neg weights (same as V24)
all_targets = []
for items, _ in user_sequences:
    all_targets.extend(items[1:])
tgt_counter = Counter(all_targets)
neg_weights = np.array([tgt_counter.get(i, 1) for i in range(NI)], dtype=np.float32)
neg_weights = np.power(neg_weights, 0.75); neg_weights /= neg_weights.sum()

# Split: user-level 80/20
n_val = N_USERS // 5
indices = list(range(N_USERS)); random.shuffle(indices)
val_users_set = set(indices[:n_val])
train_users = [i for i in indices if i not in val_users_set]

# Val: short prefixes only (1-5 items)
val_examples = []
for ui in val_users_set:
    items, u_idx = user_sequences[ui]
    for k in range(1, min(len(items), 6)):
        val_examples.append((ui, items[:k], items[k]))
print(f'  Train users: {len(train_users)}, Val examples: {len(val_examples)}')

# ── Model ──
class SASRec(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_emb = nn.Embedding(NI, EMB_DIM, padding_idx=-1)
        self.pos_emb = nn.Embedding(MAX_LEN, EMB_DIM)
        self.emb_dropout = nn.Dropout(DROPOUT)
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.attn_ln = nn.ModuleList()
        self.ffn_ln = nn.ModuleList()
        for _ in range(N_LAYERS):
            self.attn_layers.append(nn.MultiheadAttention(EMB_DIM, N_HEADS, dropout=DROPOUT, batch_first=True))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(EMB_DIM, EMB_DIM*4), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(EMB_DIM*4, EMB_DIM), nn.Dropout(DROPOUT)))
            self.attn_ln.append(nn.LayerNorm(EMB_DIM))
            self.ffn_ln.append(nn.LayerNorm(EMB_DIM))
        self.user_proj = nn.Linear(NF_USER, EMB_DIM) if NF_USER > 0 else None
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def get_attn_mask(self, L):
        return torch.triu(torch.ones(L, L, device=DEV), diagonal=1).bool()

    def forward(self, seqs, uf=None):
        B, L = seqs.shape
        emb = self.item_emb(seqs)
        pos = torch.arange(L, device=DEV).unsqueeze(0).expand(B, -1).clamp(max=MAX_LEN-1)
        emb = emb + self.pos_emb(pos); emb = self.emb_dropout(emb)
        am = self.get_attn_mask(L); kp = (seqs == 0)
        for i in range(N_LAYERS):
            ao, _ = self.attn_layers[i](emb, emb, emb, attn_mask=am, key_padding_mask=kp)
            emb = self.attn_ln[i](emb + ao)
            emb = self.ffn_ln[i](emb + self.ffn_layers[i](emb))
        lengths = (~kp).sum(1) - 1; lengths = lengths.clamp(min=0)
        ur = emb[torch.arange(B), lengths]
        if self.user_proj is not None and uf is not None:
            ur = ur + self.user_proj(uf.to(DEV))
        return ur @ self.item_emb.weight.T

# ── Train  ──
for seed_idx, SEED in enumerate(SEEDS):
    print(f'\n{"="*60}')
    print(f'SEED {SEED} ({seed_idx+1}/{len(SEEDS)})')
    print(f'{"="*60}')
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model = SASRec().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

    best_val = 0; best_state = None; no_improve = 0
    t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        total_loss, nb = 0, 0
        random.shuffle(train_users)
        for b_start in range(0, len(train_users), BATCH):
            b_end = min(b_start + BATCH, len(train_users))
            b_users = train_users[b_start:b_end]
            prefixes, tgts, uf_list, max_l = [], [], [], 0
            for ui in b_users:
                items, u_idx = user_sequences[ui]
                k = random.randint(1, len(items) - 1)
                pref = items[:k]; tgt = items[k]
                prefixes.append(pref); tgts.append(tgt)
                max_l = max(max_l, len(pref))
                uf_list.append(user_feat_n[u_idx])

            bs = len(prefixes)
            pad = torch.zeros(bs, max_l, dtype=torch.long, device=DEV)
            for i, pref in enumerate(prefixes):
                pad[i, :len(pref)] = torch.tensor(pref, dtype=torch.long)
            uf_t = torch.tensor(np.array(uf_list), dtype=torch.float32, device=DEV)
            scores = model(pad, uf_t)

            pos = torch.tensor(tgts, dtype=torch.long, device=DEV)
            pos_s = scores[torch.arange(bs), pos]
            neg = np.random.choice(NI, size=(bs, NEG), p=neg_weights)
            neg_t = torch.tensor(neg, dtype=torch.long, device=DEV)
            neg_s = scores[torch.arange(bs).unsqueeze(1), neg_t]
            loss = -F.logsigmoid(pos_s.unsqueeze(1) - neg_s).mean()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item(); nb += 1
        scheduler.step()

        # Validate
        if len(val_examples) > 0 and (ep % 5 == 0 or ep == EPOCHS-1):
            model.eval()
            val_hits = 0
            with torch.no_grad():
                for ui, pref, tgt in val_examples[:5000]:
                    items, u_idx = user_sequences[ui]
                    seq_t = torch.tensor([pref], dtype=torch.long, device=DEV)
                    uf_t = torch.tensor(user_feat_n[u_idx:u_idx+1], dtype=torch.float32, device=DEV) if u_idx is not None else None
                    sc = model(seq_t, uf_t).squeeze(0).cpu().numpy()
                    val_hits += int(np.argmax(sc) == tgt)
            val_acc = val_hits / min(5000, len(val_examples))
            elapsed = time.time() - t0
            print(f'  Ep{ep:3d} loss={total_loss/nb:.4f} val_hit={val_acc:.4f} best={best_val:.4f} {elapsed:.0f}s', flush=True)

            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 20:
                    print(f'  Early stop at ep {ep}')
                    break

    torch.save(best_state, os.path.join(OUT, f'v31_seed{SEED}.pt'))
    print(f'  Saved v31_seed{SEED}.pt (val={best_val:.4f})')

print(f'\n{"="*60}')
print('ALL DONE')
print(f'{"="*60}')
