import sys
sys.path.append('.')
import time
import threading
from modules.distill import Distiller, DataConf, TrainConf, DistillConf, NetworkConf

if __name__ == '__main__':
    data_cfg = DataConf()
    train_cfg = TrainConf()
    distill_cfg = DistillConf()
    network_cfg = NetworkConf()
    distiller = Distiller(data_cfg, train_cfg, distill_cfg, network_cfg, save_dir='/data2_hdd/xh/dis/distill2/painnasprinbase1')

    distill_done = threading.Event()

    def on_pipeline_done():
        print('Distiller pipeline finished!')
        distill_done.set()

    distiller(
        non_blocking=True,
        num_expert_trajectory=1,
        gpu_ids_list=[6],
        done_callback=on_pipeline_done
    )
    print('Executing non-blocking distillation...')

    while not distill_done.is_set():
        print("Main thread: Waiting for distillation to finish...")
        time.sleep(5)

    print("Main thread: Detected distillation completion!")
