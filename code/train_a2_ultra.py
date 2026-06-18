"""A2 ULTRA — SASRec × 5 seed ensemble, ItemCF, V100 16GB full burn.

Usage: python3 train_a2_ultra.py data/rec_data /tmp/a2_out
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[A2_ULTRA] Device: {DEV}", flush=True)

# ═══════ Heavy SASRec — 4 layers × 8 heads × 256dim ═══════
class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim=256, max_len=50, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 2, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, seq):
        B, L = seq.shape
        causal_mask = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.ln(self.drop(x))
        logits = x @ self.item_emb.weight[1:].T
        return logits

# ═══════ Data ═══════
def load_and_build(data_dir, max_len=50):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    item_df = pd.read_csv(os.path.join(data_dir, 'item.csv'))
    all_iids = sorted(item_df['iid'].tolist())
    iid2idx = {iid: i + 1 for i, iid in enumerate(all_iids)}
    idx2iid = {i + 1: iid for i, iid in enumerate(all_iids)}
    NI = len(all_iids)

    def parse_seq(seq_str):
        if pd.isna(seq_str) or not seq_str: return []
        items = [int(x) for x in str(seq_str).split(',') if x.strip()]
        return [iid2idx[i] for i in items if i in iid2idx]

    train_data = []
    for _, row in train_df.iterrows():
        seq = parse_seq(row.get('item_seq_dedup', row.get('item_seq_raw', '')))
        if len(seq) < 2: continue
        for i in range(1, len(seq)):
            prefix = seq[max(0, i - max_len + 1):i]
            train_data.append((prefix, seq[i]))

    test_data = []
    for _, row in test_df.iterrows():
        uid = int(row['uid'])
        seq = parse_seq(row.get('item_seq_dedup', row.get('item_seq_raw', '')))
        test_data.append((uid, seq[-max_len + 1:] if len(seq) > max_len else seq))
    return train_data, test_data, NI, iid2idx, idx2iid

class SeqDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.data, self.max_len = data, max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        seq, target = self.data[idx]
        seq = seq[-self.max_len + 1:]
        padded = [0] * (self.max_len - len(seq) - 1) + seq + [target]
        return torch.tensor(padded, dtype=torch.long)

# ═══════ ItemCF ═══════
def build_itemcf(train_csv, iid2idx, NI):
    df = pd.read_csv(train_csv)
    cooc = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        seq_str = str(row.get('item_seq_dedup', row.get('item_seq_raw', '')))
        items = [int(x) for x in seq_str.split(',') if x.strip()]
        idxs = [iid2idx[i] for i in items if i in iid2idx]
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                cooc[idxs[i]][idxs[j]] += 1; cooc[idxs[j]][idxs[i]] += 1
    sim = np.zeros((NI + 2, NI + 2), dtype=np.float32)
    for i, neighbors in cooc.items():
        norm = sum(neighbors.values())
        if norm > 0:
            for j, cnt in neighbors.items(): sim[i][j] = cnt / norm
    return sim

# ═══════ Train one seed ═══════
def train_seed(seed, train_data, NI, max_len, embed_dim, num_heads, num_layers, dropout, lr, wd, epochs, batch_size, val_ratio=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_data))
    vs = int(len(train_data) * val_ratio)
    trn_sub = [train_data[i] for i in perm[vs:]]
    val_sub = [train_data[i] for i in perm[:vs]]

    trn_ds = SeqDataset(trn_sub, max_len); val_ds = SeqDataset(val_sub, max_len)
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SASRec(NI, embed_dim=embed_dim, max_len=max_len,
                   num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=epochs // 4, T_mult=2)

    best_val_loss = float('inf'); best_state = None
    val_mrr = 0.0

    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for st in trn_loader:
            st = st.to(DEV); logits = model(st)
            pred, tgt = logits[:, :-1, :], st[:, 1:]
            valid = (tgt != 0); total = valid.sum()
            if total == 0: continue
            loss = F.cross_entropy(pred[valid], tgt[valid].long(), reduction='mean')
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        sched.step(); tl /= max(len(trn_loader), 1)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for st in val_loader:
                st = st.to(DEV); logits = model(st)
                pred, tgt = logits[:, :-1, :], st[:, 1:]
                valid = (tgt != 0); total = valid.sum()
                if total > 0:
                    vl += F.cross_entropy(pred[valid], tgt[valid].long(), reduction='sum').item() / total.item()

        # MRR
        v_mrr = 0.0; v_cnt = 0
        with torch.no_grad():
            for st in val_loader:
                st = st.to(DEV); logits = model(st)[:, -2, :]
                tgt = st[:, -1]; valid = (tgt != 0)
                if valid.sum() == 0: continue
                for i in range(len(st)):
                    if not valid[i]: continue
                    _, top10 = logits[i].topk(min(10, NI))
                    top10 = (top10 + 1).cpu().numpy()
                    if tgt[i].item() in top10:
                        v_mrr += 1.0 / (list(top10).index(tgt[i].item()) + 1)
                    v_cnt += 1
        if v_cnt > 0: v_mrr /= v_cnt

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            val_mrr = v_mrr

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"  Ep{ep:3d}: loss={tl:.4f} val_loss={vl:.4f} mrr={v_mrr:.4f} lr={opt.param_groups[0]['lr']:.6f}", flush=True)

    model.load_state_dict(best_state)
    return model, val_mrr

# ═══════ Main ═══════
def main():
    if len(sys.argv) < 3:
        print("Usage: python3 train_a2_ultra.py <data_dir> <output_dir>")
        sys.exit(1)
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # ULTRA hyperparams
    MAX_LEN, EMBED_DIM, NUM_HEADS, NUM_LAYERS = 50, 256, 8, 4
    DROPOUT, LR, WD, EPOCHS, BS = 0.1, 0.001, 1e-4, 80, 1024
    N_SEEDS = 3
    SEEDS = [42, 777, 2024]

    print(f"[A2_ULTRA] Loading data...", flush=True)
    train_data, test_data, NI, iid2idx, idx2iid = load_and_build(data_dir, MAX_LEN)
    print(f"[A2_ULTRA] Items:{NI} Train_samples:{len(train_data)} Test:{len(test_data)}", flush=True)
    print(f"[A2_ULTRA] Config: dim={EMBED_DIM} heads={NUM_HEADS} layers={NUM_LAYERS} bs={BS} epochs={EPOCHS}", flush=True)

    import time; t0 = time.time()
    models, mrrs = [], []
    for i, seed in enumerate(SEEDS):
        print(f"\n── Seed {i+1}/{N_SEEDS} (seed={seed}) ──", flush=True)
        model, mrr = train_seed(seed, train_data, NI, MAX_LEN, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
                                DROPOUT, LR, WD, EPOCHS, BS)
        models.append(model); mrrs.append(mrr)
        print(f"  Seed {seed}: val_mrr={mrr:.4f}, elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\n[A2_ULTRA] Training done: avg_mrr={np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}", flush=True)

    # ═══ Ensemble inference ═══
    train_csv = os.path.join(data_dir, 'train.csv')
    sim_mat = build_itemcf(train_csv, iid2idx, NI)
    df = pd.read_csv(train_csv)
    popular = df['target_iid'].value_counts().index.tolist()[:50]
    pop_idxs = [iid2idx.get(p, 1) for p in popular if p in iid2idx]

    results = []
    for uid, seq in test_data:
        scores = np.zeros(NI)
        if len(seq) == 0:
            candidates = pop_idxs[:10]
        else:
            seq_t = torch.tensor([seq[-MAX_LEN + 1:]], dtype=torch.long, device=DEV)
            for model in models:
                model.eval()
                with torch.no_grad():
                    logits = model(seq_t)
                    scores += logits[0, -1, :].cpu().numpy()
            scores /= N_SEEDS

            top100 = np.argsort(scores)[::-1][:100] + 1
            last_item = seq[-1] if seq[-1] < len(sim_mat) else 1
            cf_neighbors = np.argsort(sim_mat[last_item])[::-1][:20]
            cf_candidates = [c for c in cf_neighbors if 0 < c <= NI]

            candidates = list(top100[:5])
            for c in cf_candidates:
                if c not in candidates: candidates.append(c)
                if len(candidates) >= 10: break
            while len(candidates) < 10:
                c = pop_idxs[len(candidates) % len(pop_idxs)]
                if c not in candidates: candidates.append(c)

        pred_iids = [str(idx2iid.get(c, str(c))) for c in candidates[:10]]
        results.append({'uid': uid, 'prediction': ','.join(pred_iids)})

    out_df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, 'A2.csv')
    out_df.to_csv(out_path, index=False)

    info = {'seeds': mrrs, 'avg_mrr': float(np.mean(mrrs)), 'std_mrr': float(np.std(mrrs)),
            'NI': NI, 'train_samples': len(train_data), 'test': len(test_data),
            'config': f'dim={EMBED_DIM} heads={NUM_HEADS} layers={NUM_LAYERS} bs={BS} epochs={EPOCHS}',
            'time_min': f'{(time.time()-t0)/60:.1f}', 'output': out_path}
    with open(os.path.join(out_dir, 'a2_ultra_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[RESULT] A2 ULTRA: {N_SEEDS}seeds | "
          f"avg_mrr={np.mean(mrrs):.4f} ± {np.std(mrrs):.4f} | "
          f"dim={EMBED_DIM} L={NUM_LAYERS} H={NUM_HEADS} bs={BS} ep={EPOCHS} | "
          f"{(time.time()-t0)/60:.1f}min", flush=True)
    print(f"Output: {out_path}", flush=True)

if __name__ == '__main__':
    main()
