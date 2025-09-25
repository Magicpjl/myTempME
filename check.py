import h5py
import numpy as np
import os.path as osp

# ======= 配置项 =======
data = "wikipedia"
base_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'processed')
node_embed_size = 9228
edge_embed_size = 157474

# ======= 加载文件 =======
print("📦 加载 H5 和 NPY 文件...")
train_file = h5py.File(osp.join(base_dir, f"{data}_train_cat.h5"), 'r')
test_file = h5py.File(osp.join(base_dir, f"{data}_test_cat.h5"), 'r')

train_edge = np.load(osp.join(base_dir, f"{data}_train_edge.npy"))
test_edge = np.load(osp.join(base_dir, f"{data}_test_edge.npy"))

# ======= 检查函数 =======
def check_walk(walk, label):
    walk = walk[:, :, :15]  # (N, n_walks, 15)
    node_ids = walk[:, :, :6]
    edge_ids = walk[:, :, 6:9]

    print(f"\n======= 检查 {label} =======")
    print(f"🔥 {label} 节点范围: min={node_ids.min()}, max={node_ids.max()}")
    print(f"🔥 {label} 边范围:   min={edge_ids.min()}, max={edge_ids.max()}")

    if node_ids.min() < 0 or node_ids.max() >= node_embed_size:
        print(f"❌ {label} 节点越界！合法范围: [0, {node_embed_size - 1}]")
    else:
        print(f"✅ {label} 节点正常")

    if edge_ids.min() < 0 or edge_ids.max() >= edge_embed_size:
        print(f"❌ {label} 边越界！合法范围: [0, {edge_embed_size - 1}]")
    else:
        print(f"✅ {label} 边正常")

# ======= 检查所有 walk =======
check_walk(train_file["walks_src_new"][:], "train_src")
check_walk(train_file["walks_tgt_new"][:], "train_tgt")
check_walk(train_file["walks_bgd_new"][:], "train_bgd")

check_walk(test_file["walks_src_new"][:], "test_src")
check_walk(test_file["walks_tgt_new"][:], "test_tgt")
check_walk(test_file["walks_bgd_new"][:], "test_bgd")

# ======= 检查边文件维度 =======
print("\n======= 检查 edge.npy 对齐 =======")
print(f"train_edge shape: {train_edge.shape}")
print(f"test_edge shape: {test_edge.shape}")

for name, walk in [("train", train_file["walks_src_new"][:]),
                   ("test", test_file["walks_src_new"][:])]:
    if walk.shape[0] != (train_edge.shape[1] if name == "train" else test_edge.shape[1]):
        print(f"❌ {name} walks 和 edge.npy 行数不匹配！")
    else:
        print(f"✅ {name} walks 和 edge.npy 匹配")

# ======= 关闭文件 =======
train_file.close()
test_file.close()
