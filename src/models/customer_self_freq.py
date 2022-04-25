import numpy as np
import pandas as pd
import time
import os
import json

from tqdm import tqdm






#================================================
# Config
#================================================

args = {

    # Dataset
    "transaction"       : 'D:/workspace/Kaggle/HM_Recommend/Data/transactions_after_may.csv',
    "article"           : 'D:/workspace/Kaggle/HM_Recommend/Data/articles.csv',
    "data"              : 'D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/data/trans_500.csv',
    "sample"            : 'D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/data/submit_ref_after_may.csv',
    "submit"            : 'D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/submission/submission_after_may.csv',

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
}

def get_most_freq_items():
    df = pd.read_csv(args["transaction"])
    return df['article_id'].value_counts()[:12].index.tolist()


MOST_REFQ = get_most_freq_items()

def generate_submission():

    cid = []
    answer = []

    batch_size = 1024

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






































