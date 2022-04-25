import numpy as np
import pandas as pd

import os
import sys
import hashlib
import time
import torch as th
import json
import gc
import pdb
from tqdm import tqdm
from datetime import datetime

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F


WORKSPACEK = 'D:/workspace/Kaggle/HM_Recommend/Data/'

args = {

    # Dataset
    "train_data":   'D:\workspace\Kaggle\HM_Recommend\Kaggle_HM\data\item_cus_train.zip',
    "sample":       'D:\workspace\Kaggle\HM_Recommend\Data\sample_submission.csv',
    "predict_data": 'D:\workspace\Kaggle\HM_Recommend\Kaggle_HM\data\item_cus_predict.zip',
    "item_data":    'D:\workspace\Kaggle\HM_Recommend\Kaggle_HM\data\item_vector_2020_05_01_42d4188d97b341c97114cf45351e38ac.csv',

    "submit": 'D:\workspace\Kaggle\HM_Recommend\Kaggle_HM\submission\sample_submission_cb_v2.csv',

    "test_data_proportion": 0.1,
    "test_splite_random": 42,

    "use_data": 20000,

    # Model

    "model_name": "/content/gdrive/MyDrive/Kaggle_HM/Kaggle_HM/model/GRU_2.pt",

    # Train

    "epoch": 50,
    "batch_size": 256,
    "lr": 1e-4,

    "threshold": 0.7,

    # Log
    "log_loss_period": 10,
    "evaluate_period": 100,

}


MOST_REFQ = [706016001, 372860002, 751471001, 599580038, 610776002, 759871002, 372860001, 610776001, 841383002, 599580052, 448509014, 783346001]


def fill_ans(target):

    sample = pd.read_csv(args["sample"])
    exist = pd.read_csv(target, index_col='customer_id')

    customers = []
    recommend = []

    for cid in tqdm(sample['customer_id']):

        customers.append(cid)

        if cid in exist.index:
            recommend.append(exist.loc[cid][0])
        else:
            recommend.append("0706016001 0372860002 0751471001 0599580038 0610776002 0759871002 0372860001 0610776001 0841383002 0599580052 0448509014 0783346001")

    res = {'customer_id': customers,
           'prediction': recommend}

    submit = pd.DataFrame(data=res)
    submit.to_csv(args["submit"], index=False)

if __name__ == '__main__':

    fill_ans("./sample_submission_cb_v2_hist.csv")


'''
    # Data Prepare
    items = pd.read_csv(args["item_data"]).values

    item_feature = items[:, :-1].astype(np.int16)
    item_feature = np.transpose(item_feature)

    item_id = items[:, -1]

    #submit = pd.read_csv(args["sample"])

    customers = []
    recommend = []

    #for cid in tqdm(submit["customer_id"][:50]):

    batch_size = 1024

    with pd.read_csv(args["predict_data"], chunksize=batch_size) as reader:

        for chunk in tqdm(reader, total=665025 // batch_size):

            customers.extend(chunk.iloc[:,-1].tolist())

            value = (1000*chunk.values[:,:-2]).astype(np.int16)

            res = np.matmul(value, item_feature)

            top = item_id[np.argpartition(res, -12)[:,-12:]]

            recommend.extend([" ".join(["0" + str(x) for x in y]) for y in top])

    res = {  'customer_id' : customers,
             'prediction'  : recommend }

    submit = pd.DataFrame(data=res)

    submit.to_csv(args["submit"], index=False)

'''