"""A1 SAGE-ONLY — GraphSAGE 11 configs, 50 voters, fast & stable.

Usage: python3 train_a1_sage_only.py <data.npz> <output_dir>
"""

import os, sys, json, argparse, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[A1_SAGE] Device: {DEV}", flush=True)

# ═══════ SAGE Model ═══════
class SAGE(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 10)  # 10 classes

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.drop(x)
        return self.fc(x)

# ═══════ Configs (11 configs, 50 voters) ═══════
CONFIGS = [
    # (hidden, layers, dropout, voters)
    (64,  2, 0.3, 5),
    (64,  2, 0.5, 5),
    (64,  3, 0.3, 5),
    (64,  3, 0.5, 5),
    (128, 2, 0.3, 5),
    (128, 2, 0.5, 5),
    (128, 3, 0.3, 5),
    (128, 3, 0.5, 5),
    (256, 2, 0.3, 5),
    (256, 2, 0.5, 5),
    (256, 3, 0.3, 5),
]

# ═══════ Main ═══════
def main():
    if len(sys.argv) < 3:
        print("Usage: python3 train_a1_sage_only.py <data.npz> <output_dir>")
        sys.exit(1)
    data_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print(f"[A1_SAGE] Loading {data_path}...", flush=True)
    data = np.load(data_path, allow_pickle=True)
    x = torch.tensor(data['x'], dtype=torch.float32)
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
    y = torch.tensor(data['y'], dtype=torch.long)
    train_mask = torch.tensor(data['train_mask'], dtype=torch.bool)
    test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)
    print(f"[A1_SAGE] Nodes:{x.shape[0]} Edges:{edge_index.shape[1]} Feat_dim:{x.shape[1]}", flush=True)

    # Move to device
    x, edge_index, y = x.to(DEV), edge_index.to(DEV), y.to(DEV)
    train_mask, test_mask = train_mask.to(DEV), test_mask.to(DEV)

    t0 = time.time()
    all_probs = []

    # Checkpoint
    ckpt_path = os.path.join(out_dir, '.ckpt.npz')

    # Resume if ckpt exists
    start_i = 0
    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path, allow_pickle=True)
        # Load existing probs
        for k in ckpt.files:
            if k.startswith('p'):
                all_probs.append(torch.tensor(ckpt[k], device=DEV))
        start_i = len(all_probs)
        print(f"[A1_SAGE] [RESUME] {start_i} voters loaded", flush=True)

    # Train each config
    voter_idx = start_i
    for cfg_idx, (hidden, layers, dropout, n_voters) in enumerate(CONFIGS):
        if cfg_idx < start_i // n_voters:
            continue  # Skip completed configs

        for v in range(n_voters):
            if voter_idx < start_i:
                voter_idx += 1
                continue

            print(f"\n── Config {cfg_idx+1}/11, Voter {v+1}/{n_voters} (total {voter_idx+1}) ──", flush=True)
            print(f"   hidden={hidden}, layers={layers}, dropout={dropout}", flush=True)

            # Train
            model = SAGE(x.shape[1], hidden, layers, dropout).to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            loss_fn = nn.CrossEntropyLoss()

            model.train()
            for ep in range(200):
                opt.zero_grad()
                out = model(x, edge_index)
                loss = loss_fn(out[train_mask], y[train_mask])
                loss.backward()
                opt.step()
                if ep % 50 == 0 or ep == 199:
                    with torch.no_grad():
                        pred = out[train_mask].argmax(1)
                        acc = (pred == y[train_mask]).float().mean().item()
                        print(f"  Ep{ep:3d}: loss={loss.item():.4f} train_acc={acc:.4f}", flush=True)

            # Predict
            model.eval()
            with torch.no_grad():
                probs = F.softmax(model(x, edge_index), dim=1)
                all_probs.append(probs.cpu())
                print(f"  Voter {voter_idx+1} done, probs shape: {probs.shape}", flush=True)

            # Save checkpoint
            ckpt_dict = {f'p{i}': p.numpy() for i, p in enumerate(all_probs)}
            ckpt_dict['elapsed'] = time.time() - t0
            np.savez(os.path.join(out_dir, '.ckpt_tmp.npz'), **ckpt_dict)
            os.replace(os.path.join(out_dir, '.ckpt_tmp.npz'), ckpt_path)
            print(f"  [CKPT] Saved {voter_idx+1} voters", flush=True)
            voter_idx += 1

    # Ensemble
    print(f"\n[A1_SAGE] Ensembling {len(all_probs)} voters...", flush=True)
    ensemble_probs = torch.stack(all_probs).mean(0)
    pred = ensemble_probs.argmax(1)

    # Save prediction
    test_idx = np.where(test_mask.cpu().numpy())[0]
    pred_labels = pred[test_mask].cpu().numpy()
    out_df = pd.DataFrame({'test_idx': test_idx, 'label': pred_labels})
    out_path = os.path.join(out_dir, 'A1.csv')
    out_df.to_csv(out_path, index=False)
    print(f"[A1_SAGE] Saved to {out_path}", flush=True)

    # Accuracy
    true_labels = y[test_mask].cpu().numpy()
    acc = (pred_labels == true_labels).mean()
    print(f"\n{'='*60}")
    print(f"[RESULT] A1 SAGE-ONLY: {len(all_probs)} voters | accuracy={acc:.4f} | {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == '__main__':
    main()
