"""A2 PURE: 纯SASRec 5种子集成，无ItemCF毒药
V24架构(val NDCG=0.7042) + 5种子BPR + 轻量Pop兜底
预计: ~30min/seed × 5 = 2.5h (RTX 5060)
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
import os, time, random, math, sys

BASE = r'C:\Users\高帅东\Desktop\tianchi_afac'
DEV = torch.device('cuda')
SEEDS = [42, 123, 456, 789, 1011]
N_NEG = 5
EMB_DIM=128; N_HEADS=4; N_LAYERS=2; DROPOUT=0.2; MAX_LEN=200
BATCH=256; EPOCHS=80; PATIENCE=15; LR=1e-3; WD=1e-4

OUT_DIR = sys.argv[1] if len(sys.argv)>1 else os.path.join(BASE,'a2_pure_out')
os.makedirs(OUT_DIR, exist_ok=True)

# ═══ Load ═══
print("[PURE] Loading...", flush=True)
train_df = pd.read_csv(os.path.join(BASE, 'A_recommend', 'train.csv'))
test_df  = pd.read_csv(os.path.join(BASE, 'A_recommend', 'test.csv'))
item_df  = pd.read_csv(os.path.join(BASE, 'A_recommend', 'item.csv'))
user_df  = pd.read_csv(os.path.join(BASE, 'A_recommend', 'user.csv'))
all_iids = item_df['iid'].tolist(); NI = len(all_iids)
iid2idx = {iid: i for i,iid in enumerate(all_iids)}
idx2iid = {i: iid for i,iid in enumerate(all_iids)}
print(f"  Items={NI}", flush=True)

def pp(s):
    if pd.isna(s) or str(s) in ('nan',''): return []
    return [x.strip() for x in str(s).split(',') if x.strip()]

# User features
uf_cols = [c for c in user_df.columns if c != 'uid']
for c in uf_cols:
    if user_df[c].dtype == 'object': user_df[c]=user_df[c].astype('category').cat.codes
    user_df[c]=user_df[c].fillna(0).astype(np.float32)
NF_USER=len(uf_cols)
uid2idx = {uid:i for i,uid in enumerate(user_df['uid'])}
uf_raw=user_df[uf_cols].values
uf_m=uf_raw.mean(0,keepdims=True); uf_s=uf_raw.std(0,keepdims=True).clip(min=1e-8)
user_feat_n = torch.tensor((uf_raw-uf_m)/uf_s, dtype=torch.float32)

# Pop ranking (for cold-start)
tgt_counts = Counter(train_df['target_iid'])
pop_ranking = [iid for iid,_ in tgt_counts.most_common()]
top10_pop = pop_ranking[:10]

# ═══ SASRec (V24 architecture) ═══
class SASRec(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_emb=nn.Embedding(NI, EMB_DIM)
        self.pos_emb=nn.Embedding(MAX_LEN, EMB_DIM)
        self.emb_drop=nn.Dropout(DROPOUT)
        self.attn_layers=nn.ModuleList(); self.ffn_layers=nn.ModuleList()
        self.attn_ln=nn.ModuleList(); self.ffn_ln=nn.ModuleList()
        for _ in range(N_LAYERS):
            self.attn_layers.append(nn.MultiheadAttention(EMB_DIM,N_HEADS,dropout=DROPOUT,batch_first=True))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(EMB_DIM,EMB_DIM*4),nn.GELU(),nn.Dropout(DROPOUT),
                nn.Linear(EMB_DIM*4,EMB_DIM),nn.Dropout(DROPOUT)))
            self.attn_ln.append(nn.LayerNorm(EMB_DIM)); self.ffn_ln.append(nn.LayerNorm(EMB_DIM))
        self.user_proj=nn.Linear(NF_USER,EMB_DIM)
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)

    def forward(self, seqs, uf, return_all=False):
        B,L=seqs.shape; emb=self.item_emb(seqs)
        pos=torch.arange(L,device=DEV).unsqueeze(0).clamp(max=MAX_LEN-1)
        emb=emb+self.pos_emb(pos); emb=self.emb_drop(emb)
        if uf is not None: emb=emb+self.user_proj(uf.to(DEV)).unsqueeze(1)
        causal=torch.triu(torch.ones(L,L,device=DEV)*float('-inf'),diagonal=1)
        pad_mask=(seqs==0)
        for i in range(N_LAYERS):
            ao,_=self.attn_layers[i](emb,emb,emb,attn_mask=causal,key_padding_mask=pad_mask)
            emb=self.attn_ln[i](emb+ao); emb=self.ffn_ln[i](emb+self.ffn_layers[i](emb))
        lengths=(~pad_mask).sum(1)-1; lengths=lengths.clamp(min=0)
        last=emb[torch.arange(B),lengths]
        scores=last@self.item_emb.weight.T
        return scores

# ═══ Build sequences ═══
print("[PURE] Building sequences...", flush=True)
user_seqs = []
for i in range(len(train_df)):
    row=train_df.iloc[i]
    raw=pp(row['item_seq_raw'])
    items=[iid2idx[iid] for iid in raw if iid in iid2idx]
    u_idx=uid2idx.get(row['uid'])
    if len(items)>=2 and u_idx is not None:
        user_seqs.append((items, u_idx))
N_USERS=len(user_seqs)
print(f"  Users={N_USERS}", flush=True)

# Split 1/7 for validation
n_val=N_USERS//7; all_idx=list(range(N_USERS)); random.shuffle(all_idx)
val_users=set(all_idx[:n_val]); train_users=[i for i in all_idx if i not in val_users]

# Neg sampling weights
all_tgts=[]
for items,_ in user_seqs: all_tgts.extend(items[1:])
tgt_counter=Counter(all_tgts)
neg_w=np.array([tgt_counter.get(i,1) for i in range(NI)],dtype=np.float32)
neg_w=np.power(neg_w,0.75); neg_w/=neg_w.sum()

# ═══ Train one seed ═══
def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model=SASRec().to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2)
    
    best_val=0; best_state=None; pc=0; s0=time.time()
    for ep in range(EPOCHS):
        model.train()
        random.shuffle(train_users); tl,nb=0,0
        for b0 in range(0,len(train_users),BATCH):
            bu=train_users[b0:b0+BATCH]; bs=len(bu)
            ps,tgs,ufs,ml=[],[],[],0
            for ui in bu:
                items,u_idx=user_seqs[ui]; items=items[-MAX_LEN:]
                k=random.randint(1,len(items)-1)
                ps.append(items[:k]); tgs.append(items[k]); ml=max(ml,k)
                ufs.append(user_feat_n[u_idx])
            pad=torch.zeros(bs,ml,dtype=torch.long,device=DEV)
            for i,p in enumerate(ps): pad[i,:len(p)]=torch.tensor(p,dtype=torch.long)
            uf_t=torch.stack([ufs[i].to(DEV) for i in range(bs)]) if bs>0 else None
            
            sc=model(pad,uf_t)
            pos_t=torch.tensor(tgs,dtype=torch.long,device=DEV)
            pos_s=sc[torch.arange(bs),pos_t]
            neg=np.random.choice(NI,size=(bs,N_NEG),p=neg_w)
            neg_t=torch.tensor(neg,dtype=torch.long,device=DEV)
            neg_s=sc[torch.arange(bs).unsqueeze(1),neg_t]
            loss=-F.logsigmoid(pos_s.unsqueeze(1)-neg_s).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step(); sched.step(); tl+=loss.item(); nb+=1
        
        # Validate (short prefix, like V24)
        if ep%5==0 or ep==EPOCHS-1:
            model.eval(); vn=0
            vl=list(val_users); random.shuffle(vl)
            with torch.no_grad():
                for vi in range(0,min(len(vl),2000),128):
                    vb=vl[vi:vi+128]; bsn=len(vb)
                    pss,tgs2,ufs2,ml2=[],[],[],0
                    for ui in vb:
                        items,u_idx=user_seqs[ui]; items=items[-MAX_LEN:]
                        k=random.randint(1,min(len(items)-1,5))
                        pss.append(items[:k]); tgs2.append(items[k]); ml2=max(ml2,k)
                        ufs2.append(user_feat_n[u_idx])
                    pad2=torch.zeros(bsn,ml2,dtype=torch.long,device=DEV)
                    for i,p in enumerate(pss): pad2[i,:len(p)]=torch.tensor(p,dtype=torch.long)
                    ufv2=torch.stack([ufs2[i].to(DEV) for i in range(bsn)])
                    sc2=model(pad2,ufv2); preds=sc2.argsort(dim=-1,descending=True)
                    for i,t in enumerate(tgs2):
                        top10=preds[i,:10].cpu().numpy()
                        if t in top10: vn+=1.0/math.log2(float(list(top10).index(t))+2)
            val_ndcg=vn/max(len(vl),1)
            if val_ndcg>best_val:
                best_val=val_ndcg; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; pc=0
            else: pc+=1
            print(f"  Seed{seed} Ep{ep:2d}: loss={tl/max(nb,1):.4f} val={val_ndcg:.4f} best={best_val:.4f} pc={pc}", flush=True)
            if pc>=PATIENCE: print(f"  Early stop@{ep}", flush=True); break
    
    model.load_state_dict(best_state); model.eval()
    print(f"  Seed{seed} done: best={best_val:.4f}, {time.time()-s0:.0f}s", flush=True)
    return model, best_val

# ═══ 5-seed ensemble ═══
models, mrrs = [], []
for seed in SEEDS:
    print(f"\n── Seed {seed} ──", flush=True)
    m, v = train_seed(seed)
    models.append(m); mrrs.append(v)

print(f"\n[PURE] Seeds done: avg={np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}", flush=True)

# ═══ Predict ═══
print("[PURE] Predicting...", flush=True)
preds = []
for _, row in test_df.iterrows():
    raw=pp(row['item_seq_raw'])
    items=[iid2idx[iid] for iid in raw if iid in iid2idx][-MAX_LEN:]
    u_idx=uid2idx.get(row['uid'])
    
    if not items:
        preds.append(','.join(top10_pop))
        continue
    
    seq_t=torch.tensor([items],dtype=torch.long,device=DEV)
    uf_t=user_feat_n[u_idx].unsqueeze(0).to(DEV) if u_idx is not None else None
    
    scores=np.zeros(NI)
    for m in models:
        with torch.no_grad():
            sc=m(seq_t,uf_t).squeeze(0).cpu().numpy()
            scores+=sc
    scores/=len(models)
    
    top10=np.argsort(-scores)[:10]
    preds.append(','.join(idx2iid[i] for i in top10))

out=pd.DataFrame({'uid':test_df['uid'],'prediction':preds})
out.to_csv(os.path.join(OUT_DIR,'A2.csv'),index=False)
uniq=set(); [uniq.update(p.split(',')) for p in preds]
print(f"[PURE] A2.csv: {len(out)} rows, {len(uniq)} products", flush=True)
print(f"[PURE] val NDGCs: {mrrs}", flush=True)
