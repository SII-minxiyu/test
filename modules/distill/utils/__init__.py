import torch.nn as nn
from .adapter_mtt import AdapterMTT
from .adapter_mtt2 import AdapterMTT2
from .adapter_mtt6 import AdapterMTT6
from .distill_without_lora import AdapterMTT3
from.adapter_mtt4 import AdapterMTT4
from .adapter_mtt5 import AdapterMTT5
from .adapter_mtt6 import AdapterMTT6
from .distill_meta import AdapterMTT7
def get_distill_algorithm(algorithm: str):
    algorithm = algorithm.lower()
    if algorithm == 'adapter_mtt':
        return AdapterMTT()
    elif algorithm == 'adapter_mtt2':
        return AdapterMTT2()
    elif algorithm == 'adapter_mtt3':
        return AdapterMTT3()
    elif algorithm == 'adapter_mtt4':
        return AdapterMTT4()
    elif algorithm == 'adapter_mtt5':
        return AdapterMTT5()
    elif algorithm == 'adapter_mtt6':
        return AdapterMTT6()
    elif algorithm == 'adapter_mtt7':
        return AdapterMTT7()
