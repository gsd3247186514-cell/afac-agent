"""A2 ItemCF: 基于物品的协同过滤，快速凑合版"""
import pandas as pd
import numpy as np
from collections import defaultdict

def load_data(data_dir):
    train_df = pd.read_csv(f'{data_dir}/train_data.csv')
    test_df = pd.read_csv(f'{data_dir}/test_data.csv')
    return train_df, test_df

def build_itemcf(train_df):
    """构建物品共现矩阵"""
    item_pairs = defaultdict(int)
    user_count = defaultdict(int)
    
    for _, row in train_df.iterrows():
        seq = [x.strip().lstrip('iI') for x in str(row['item_seq_dedup']).split(',') if x.strip()]
        seq = list(map(int, seq))
        for i in range(len(seq)):
            for j in range(i+1, len(seq)):
                item_pairs[(seq[i], seq[j])] += 1
                item_pairs[(seq[j], seq[i])] += 1
        for item in seq:
            user_count[item] += 1
    
    # 计算物品相似度
    item_sim = defaultdict(dict)
    for (i, j), cnt in item_pairs.items():
        sim = cnt / max(user_count[i], user_count[j], 1)
        item_sim[i][j] = sim
        item_sim[j][i] = sim
    
    return item_sim, user_count

def predict(test_df, item_sim, user_count, top_k=10):
    """预测测试集"""
    results = []
    for _, row in test_df.iterrows():
        uid = row['uid']
        seq = [x.strip().lstrip('iI') for x in str(row['item_seq_dedup']).split(',') if x.strip()]
        seq = list(map(int, seq))
        
        # 基于历史物品，推荐相似物品
        scores = defaultdict(float)
        for item in seq:
            if item in item_sim:
                for neighbor, sim in item_sim[item].items():
                    if neighbor not in seq:  # 不推荐已交互过的
                        scores[neighbor] += sim
        
        # 如果分数全0，推荐最受欢迎的物品
        if not scores:
            popular_items = sorted(user_count.items(), key=lambda x: -x[1])[:top_k]
            pred = [item for item, _ in popular_items]
        else:
            pred = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
            pred = [item for item, _ in pred]
        
        # 补齐到10个
        while len(pred) < top_k:
            pred.append(pred[-1])
        
        results.append([uid] + pred[:top_k])
    
    return results

def main():
    data_dir = '/mnt/workspace/framework/data/rec_data'
    out_dir = '/mnt/workspace/a2_itemcf_out'
    
    print('[A2_ItemCF] Loading data...', flush=True)
    train_df, test_df = load_data(data_dir)
    
    print('[A2_ItemCF] Building ItemCF...', flush=True)
    item_sim, user_count = build_itemcf(train_df)
    
    print('[A2_ItemCF] Predicting...', flush=True)
    results = predict(test_df, item_sim, user_count, top_k=10)
    
    # 保存结果
    import os
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results, columns=['uid'] + [f'item_{i}' for i in range(10)])
    df.to_csv(f'{out_dir}/A2.csv', index=False)
    print(f'[A2_ItemCF] Done! Result saved to {out_dir}/A2.csv', flush=True)

if __name__ == '__main__':
    main()
