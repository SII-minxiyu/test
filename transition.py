import h5py

file_path = "/data2_hdd/xh/transition1x/SPICE-1.1.2.hdf5"

with h5py.File(file_path, "r") as f:
    def explore(name, obj):
        print(name)

    print("Top-level keys:", list(f.keys()))
    print("\nExploring file structure:\n")
    f.visititems(explore)


# from transition1x import Dataloader
# import numpy as np
#
# path = "/data2_hdd/xh/transition1x/Transition1x.h5"
# loader = Dataloader(path, datasplit="train", only_final=False)
#
# positions, forces, energies, atomic_numbers, names = [], [], [], [], []
#
# for mol in loader:
#     positions.append(np.array(mol["positions"]))
#     forces.append(np.array(mol["wB97x_6-31G(d).forces"]))
#     energies.append(mol["wB97x_6-31G(d).energy"])
#     atomic_numbers.append(np.array(mol["atomic_numbers"]))
#     names.append(mol["rxn"])
#
# np.savez(
#     "transition1x_md.npz",
#     positions=np.array(positions, dtype=object),
#     forces=np.array(forces, dtype=object),
#     energy=np.array(energies),
#     atomic_numbers=np.array(atomic_numbers, dtype=object),
#     names=np.array(names)
# )
#
# print("✅ Saved as transition1x_md.npz")

# from transition1x import Dataloader
# from collections import defaultdict
# from tqdm import tqdm  # ✅ 进度条
# import os
#
# # === 配置 ===
# h5_path = "/data2_hdd/xh/transition1x/Transition1x.h5"
# datasplit = "train"   # 可选: "data", "train", "val", "test"
#
# # === 初始化 ===
# loader = Dataloader(h5_path, datasplit=datasplit, only_final=False)
#
# all_formulas = set()
# count_by_rxn = defaultdict(int)
#
# # === 遍历 + 进度条 ===
# print(f"🔍 Scanning {datasplit} split in {os.path.basename(h5_path)} ...")
#
# for mol in tqdm(loader, desc="Reading molecules", ncols=100):
#     formula = mol["formula"]
#     rxn = mol["rxn"]
#     all_formulas.add(formula)
#     count_by_rxn[rxn] += 1
#
# # === 输出结果 ===
# print("\n🧪 分子体系（formula）:")
# for f in sorted(all_formulas):
#     print("  -", f)
#
# print("\n⚗️ 反应体系（rxn）及样本数:")
# for rxn, count in sorted(count_by_rxn.items(), key=lambda x: x[0]):
#     print(f"  - {rxn:<35} {count:>6} frames")

