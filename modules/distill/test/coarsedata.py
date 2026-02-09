from modules.distill.utils.datasets import DistillDataset, save_dataset
import os
data_path ='/home/xh/project/distill/modules/distill/test/benzene_test_data.pt'


dataset = DistillDataset.from_pt(data_path=data_path, use_coarse_label=True)

save_dataset(dataset, os.path.join('/home/xh/project/distill/modules/distill/test', 'syn_test_dataset.pt'))