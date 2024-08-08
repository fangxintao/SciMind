import mindspore.dataset as ds
import numpy as np

np.set_printoptions(threshold=np.inf)

mindrecord_files = ["/home/ma-user/work/r0.8_fangxt/htc_data/smiles_alpaca_all_2048/smiles_alpaca.mindrecord0"] # contains 1 or multiple MindRecord files
dataset = ds.MindDataset(dataset_files=mindrecord_files)

import time
for x in dataset:
    for v in x: 
        print(v.asnumpy().astype(np.int32))
    break
    # time.sleep(3)