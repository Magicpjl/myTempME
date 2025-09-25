import pandas as pd
import numpy as np
import os
import shutil

# ====== 设定路径 ======
source_wiki_path = '/home/userdata/magicpjl/fact8-temporal-graph/tgnnexplainer/xgraph/dataset/data/wikipedia.csv'
tempme_root = '/home/userdata/magicpjl/TempME'
processed_dir = os.path.join(tempme_root, 'processed', 'wikipedia')

# ====== 第一步：建目录 ======
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# ====== 第二步：处理 edges.csv 和 node_features.csv ======
print("爷开始处理 edges.csv 和 node_features.csv ！")

df = pd.read_csv(source_wiki_path)

# 生成 edges.csv
edges = df[['u', 'i', 'ts']].copy()
edges['edge_idx'] = range(len(edges))
edges.columns = ['source', 'destination', 'timestamp', 'edge_idx']
edges.to_csv(os.path.join(processed_dir, 'edges.csv'), index=False)

# 生成 node_features.csv
node_features = df.drop(columns=['i', 'ts', 'label']).drop_duplicates('u').sort_values('u')
node_features = node_features.set_index('u')
node_features.to_csv(os.path.join(processed_dir, 'node_features.csv'), header=False)

print("edges.csv 和 node_features.csv 已经整好了✅")

# ====== 第三步：提示下一步跑TempME自带的子图采样 ======
print("✅ edges 和 node_features处理完成！")
print()
print("下一步：切到TempME目录，跑子图预处理！")
print()
print("进入 TempME 目录:")
print(f"cd {tempme_root}")
print()
print("跑基座模型（比如TGAT）：")
print("python learn_base.py --base_type tgat --data Wikipedia")
print()
print("跑解释器（TempME motif discovery）：")
print("python temp_exp_main.py --base_type tgat --data Wikipedia")
print()
print("🐲 爷已经给你铺好路了，起飞吧！！！")
