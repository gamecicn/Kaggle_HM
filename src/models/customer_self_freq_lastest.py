import numpy as np
import pandas as pd
import time
import os
import json

from tqdm import tqdm

MOST_REFQ = ["0706016001", "0706016002",
             "0372860001", "0610776002",
             "0759871002", "0464297007",
             "0372860002", "0610776001",
             "0399223001", "0706016003",
             "0720125001", "0156231001"]



#================================================
# Config
#================================================

args = {

    # Dataset
    "article"           : 'D:/workspace/Kaggle/HM_Recommend/Data/articles.csv',
    "data"              : 'D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/data/trans_500.csv',
    "sample"            : 'D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/data/submit_ref.csv',
    "submit"            : './sample_submission.csv',

    "item_size"         : 105542,  # Fixed for HM  data set
    "pad"               : 0,

    "train_share"       : 0.8,  # training data share from whole data

    "rp_buffer_size"    : 16,  # replay buffer

    # Model
    "gru_layer"         : 1,
    "hidden_factor"     : 64,
    "state_len"         : 8,
    "model_name"        : "../model/GRU_1.pt",

    # Train
    "epoch"             : 1,
    "epoch_size"        : -1,
    "batch_size"        : 256,
    "lr"                : 0.01,

    # Log
    "log_loss_period"   : 10,
    "evaluate_period"   : 100,

    # Evaluate

    # Reward
    "r_click"           : 1.0,
    "r_buy"             : 1.0, # There is no buy in HM data set

    # Misc

    "begin_date"        : "2020-04-22"
}


def generate_submission():

    cid = []
    answer = []

    batch_size = 512

    with pd.read_csv(args["sample"], chunksize=batch_size) as reader:

        for chunk in tqdm(reader, total= 1371980//batch_size):

            cid.extend(chunk['customer_id'].tolist())

            batch = [json.loads(x) for x in chunk["article_id"]]

            batch = [pd.Series(x).value_counts()[:12].index.tolist() for x in batch]

            batch = [x + list(filter(lambda y: y not in x , MOST_REFQ))[:12 - len(x)] for x in batch]

            ans12 = [" ".join(["0" + str(y) for y in x]) for x in batch]
            answer.extend(ans12)

    res = {
        'customer_id' : cid,
        'prediction' : answer
    }

    submit = pd.DataFrame(data=res)

    submit.to_csv(args["submit"], index=False)


if __name__ == '__main__':


    generate_submission()






































