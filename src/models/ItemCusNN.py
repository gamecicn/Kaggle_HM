import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

#================================================
# Config
#================================================

args = {

    # Dataset
    "transaction"       : 'D:/workspace/Kaggle/HM_Recommend/Data/transactions_train.csv',
    "article"           : 'D:/workspace/Kaggle/HM_Recommend/Data/articles.csv',
    "customer"          : 'D:/workspace/Kaggle/HM_Recommend/Data/customers.csv',
    "sample"            : 'D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/data/submit_ref_after_may.csv',
    "submit"            : 'D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/submission/submission_after_may.csv',

    # Output
    "meta_file"         : "D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/data/item_vector",


    # Data
    "data_start"        : "2020-05-01",

    #"item_cols"         : ["product_type_no", "colour_group_code",
    #                       "perceived_colour_value_id", "perceived_colour_master_id",
    #                       "department_no", "index_code", "index_group_no", "section_no", "garment_group_no"],

    "item_cols"         : ["product_type_no", "colour_group_code",
                           "perceived_colour_value_id", "perceived_colour_master_id",
                           "index_code", "index_group_no", "section_no", "garment_group_no"],

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


}


#================================================
# Utility
#================================================

def missing_value(data):
    mis_data = data.isnull().sum().sort_values(ascending=False)
    per_data = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)
    #nunique_data = mis_data.nunique_data()
    ret_data = pd.concat([per_data,mis_data],axis=1,keys=["Percentage Missing","Missing Count"])
    return ret_data[ret_data["Missing Count"]>0] if ret_data[ret_data["Missing Count"]>0].shape[0]>0 else "No missing Value"

def unique_data(data):
    tot = data.count()
    nunique = data.nunique().sort_values()
    #perc = (data.nunique()/data.count()*100).sort_values(ascending=False)
    ret_data =  pd.concat([tot,nunique],keys=["Total","Unique Values"],axis=1)
    return ret_data.sort_values(by="Unique Values")

def item_to_vector(item_id):
    pass


import os
import hashlib


def gen_items_to_vec_files():

    col_hash = hashlib.md5(("".join(args["item_cols"])).encode("utf-8")).hexdigest()

    article_file_name = "%s_%s_%s.csv" % (args['meta_file'],
                                            args['data_start'].replace("-", "_"),
                                            col_hash)

    if os.path.exists(article_file_name):
        print("Item vector file already exist.")
        return
    else:
        print("Item vector file not exist, generate")

    # filter useless items
    trans = pd.read_csv(args["transaction"])
    trans = trans[trans["t_dat"] > args["data_start"]]
    used_items = trans["article_id"].unique()

    item_df = pd.read_csv(args["article"])
    item_df = item_df[item_df["article_id"].isin(used_items)]
    item_df = item_df[["article_id"] + args["item_cols"]]

    #
    del trans
    del used_items

    print(f"There are 105542 items originally, {len(item_df)} left after filtering.")

    #item_df.to_csv("temp.csv", index=False)
    #item_df = pd.read_csv("temp.csv")

    print(item_df.info())
    print(unique_data(item_df))

    enc = OneHotEncoder()
    enc.fit(item_df.iloc[:,1:])
    df = pd.DataFrame(enc.transform(item_df.iloc[:,1:]).toarray().astype(np.int32))
    df['article_id'] = item_df['article_id']

    df.to_csv(article_file_name, index=False)





if __name__ == '__main__':


    gen_items_to_vec_files()
