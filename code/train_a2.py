"""A2 SASRec训练脚本 — 独立工作, 不依赖Agent管线bug.

Agent Experiment阶段直接调用此脚本, 绕过train.py的Task2实现问题。
"""
import os, sys, math, json, argparse, logging
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════ 轻量SASRec ═══════
class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim=128, max_len=50, num_heads=4, num_layers=2, dropout=0.2):
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
        logits = x @ self.item_emb.weight[1:].T  # (B, L, NI)
        return logits

# ═══════ 数据加载 ═══════
def load_and_build(data_dir, max_len=50):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    item_df = pd.read_csv(os.path.join(data_dir, 'item.csv'))

    all_iids = sorted(item_df['iid'].tolist())
    iid2idx = {iid: i + 1 for i, iid in enumerate(all_iids)}  # 1-based, 0=pad
    idx2iid = {i + 1: iid for i, iid in enumerate(all_iids)}
    NI = len(all_iids)

    def parse_seq(seq_str):
        if pd.isna(seq_str) or not seq_str:
            return []
        items = [int(x) for x in str(seq_str).split(',') if x.strip()]
        return [iid2idx[i] for i in items if i in iid2idx]

    # 训练集: (seq, target)
    train_data = []
    for _, row in train_df.iterrows():
        seq = parse_seq(row.get('item_seq_dedup', row.get('item_seq_raw', '')))
        if len(seq) < 2:
            continue
        # 每个用户产生多个训练样本: [a]→b, [a,b]→c, ...
        for i in range(1, len(seq)):
            prefix = seq[max(0, i - max_len + 1):i]  # 最近max_len-1个
            target = seq[i]
            train_data.append((prefix, target))

    # 测试集: (uid, seq)
    test_data = []
    for _, row in test_df.iterrows():
        uid = int(row['uid'])
        seq = parse_seq(row.get('item_seq_dedup', row.get('item_seq_raw', '')))
        test_data.append((uid, seq[-max_len + 1:] if len(seq) > max_len else seq))

    return train_data, test_data, NI, iid2idx, idx2iid, item_df

class SeqDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        seq = seq[-self.max_len + 1:]  # 留一个位置给target
        padded = [0] * (self.max_len - len(seq) - 1) + seq + [target]
        return torch.tensor(padded, dtype=torch.long)

# ═══════ ItemCF候选生成 ═══════
def build_itemcf(train_df, iid2idx, NI):
    cooc = defaultdict(lambda: defaultdict(int))
    for _, row in train_df.iterrows():
        seq_str = str(row.get('item_seq_dedup', row.get('item_seq_raw', '')))
        items = [int(x) for x in seq_str.split(',') if x.strip()]
        idxs = [iid2idx[i] for i in items if i in iid2idx]
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                cooc[idxs[i]][idxs[j]] += 1
                cooc[idxs[j]][idxs[i]] += 1

    sim = np.zeros((NI + 2, NI + 2), dtype=np.float32)
    for i, neighbors in cooc.items():
        norm = sum(neighbors.values())
        if norm > 0:
            for j, cnt in neighbors.items():
                sim[i][j] = cnt / norm
    return sim

# ═══════ 训练 ═══════
def train(args):
    print(f"[A2] Loading data from {args.data_path}...", flush=True)
    train_data, test_data, NI, iid2idx, idx2iid, item_df = load_and_build(args.data_path, max_len=args.max_len)

    # 划分训练/验证 (90/10)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_data))
    val_size = int(len(train_data) * 0.1)
    trn_idx = perm[val_size:]
    val_idx = perm[:val_size]
    trn_sub = [train_data[i] for i in trn_idx]
    val_sub = [train_data[i] for i in val_idx]

    print(f"[A2] Train: {len(trn_sub)}, Val: {len(val_sub)}, Items: {NI}", flush=True)

    trn_ds = SeqDataset(trn_sub, max_len=args.max_len)
    val_ds = SeqDataset(val_sub, max_len=args.max_len)
    trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SASRec(NI, embed_dim=args.embedding_dim, max_len=args.max_len,
                    num_heads=args.num_heads, num_layers=args.num_layers,
                    dropout=args.dropout).to(DEV)
    print(f"[A2] SASRec {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.epochs // 4, T_mult=2)

    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_mrr': [], 'lr': []}

    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for st in trn_loader:
            st = st.to(DEV)
            logits = model(st)
            pred, tgt = logits[:, :-1, :], st[:, 1:]
            valid = (tgt != 0)
            total = valid.sum()
            if total == 0:
                continue
            loss = F.cross_entropy(pred[valid].float(), tgt[valid], reduction='mean')
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        sched.step()
        tl /= max(len(trn_loader), 1)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for st in val_loader:
                st = st.to(DEV)
                logits = model(st)
                pred, tgt = logits[:, :-1, :], st[:, 1:]
                valid = (tgt != 0)
                total = valid.sum()
                if total > 0:
                    vl += F.cross_entropy(pred[valid].float(), tgt[valid], reduction='sum').item() / total.item()

        # MRR on validation
        v_mrr = 0.0
        v_cnt = 0
        with torch.no_grad():
            for st in val_loader:
                st = st.to(DEV)
                logits = model(st)[:, -2, :]  # 最后一个有效位置
                tgt = st[:, -1]
                valid = (tgt != 0)
                if valid.sum() == 0:
                    continue
                for i in range(len(st)):
                    if not valid[i]:
                        continue
                    scores = logits[i]
                    target = tgt[i].item()
                    _, top10 = scores.topk(min(10, NI))
                    top10 = (top10 + 1).cpu().numpy()  # +1 because 0-based scores
                    if target in top10:
                        rank = list(top10).index(target) + 1
                        v_mrr += 1.0 / rank
                    v_cnt += 1

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_mrr'].append(v_mrr / max(v_cnt, 1))
        history['lr'].append(opt.param_groups[0]['lr'])

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % max(1, args.epochs // 10) == 0:
            print(f"  Ep{ep:3d}: loss={tl:.4f} val_loss={vl:.4f} mrr={v_mrr/max(v_cnt,1):.4f} lr={opt.param_groups[0]['lr']:.6f}", flush=True)

    # 保存checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': best_state,
        'args': vars(args),
        'iid2idx': iid2idx, 'idx2iid': idx2iid,
        'NI': NI, 'history': history,
    }, ckpt_path)
    print(f"[A2] Checkpoint saved: {ckpt_path}", flush=True)

    # 保存metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(history, f)
    print(f"[A2] Metrics saved: {metrics_path}", flush=True)

    # 推理生成A2.csv
    model.load_state_dict(best_state)
    model.eval()

    # Build ItemCF sim for cold-start candidates
    train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    sim_mat = build_itemcf(train_df, iid2idx, NI)
    popular = train_df['target_iid'].value_counts().index.tolist()[:50]
    pop_idxs = [iid2idx.get(p, 1) for p in popular if p in iid2idx]

    results = []
    for uid, seq in test_data:
        if len(seq) == 0:
            # 冷启动: 用流行度
            candidates = pop_idxs[:10]
        else:
            seq_t = torch.tensor([seq[-args.max_len + 1:]], dtype=torch.long, device=DEV)
            with torch.no_grad():
                logits = model(seq_t)
                scores = logits[0, -1, :].cpu().numpy()  # last position
            top100 = np.argsort(scores)[::-1][:100] + 1  # 0-based→1-based

            # ItemCF增强: 取最后一个物品的协同过滤邻居
            last_item = seq[-1] if seq[-1] < len(sim_mat) else 1
            cf_neighbors = np.argsort(sim_mat[last_item])[::-1][:20]
            cf_candidates = [c for c in cf_neighbors if c > 0 and c <= NI]

            # 合并: SASRec Top-K + ItemCF候选
            candidates = list(top100[:5])  # top 5 from SASRec
            for c in cf_candidates:
                if c not in candidates and c > 0:
                    candidates.append(c)
                if len(candidates) >= 10:
                    break
            while len(candidates) < 10:
                c = pop_idxs[len(candidates) % len(pop_idxs)]
                if c not in candidates:
                    candidates.append(c)

        pred_iids = [str(idx2iid.get(c, str(c))) for c in candidates[:10]]
        results.append({'uid': uid, 'prediction': ','.join(pred_iids)})

    out_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, 'A2.csv')
    out_df.to_csv(out_path, index=False)
    print(f"[A2] Output: {out_path} ({len(results)} rows)", flush=True)

    return 0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='task2')
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='./output')
    p.add_argument('--model_type', type=str, default='sasrec')
    p.add_argument('--embedding_dim', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()

if __name__ == '__main__':
    args, _ = parse_args().parse_known_args()  # 忽略Agent传来的未知参数
    ret = train(args)
    sys.exit(ret)
