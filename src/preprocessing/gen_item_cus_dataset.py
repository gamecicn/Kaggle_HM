import numpy as np
import pandas as pd

import os
import hashlib
import time

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

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
    "item_vector_data"         : "D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/data/item_vector",
    "cus_vector_data"          : "D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/data/cus_vector",
    "item_cus_train_data"      : "D:/workspace/Kaggle/HM_Recommend/Kaggle_HM/data/item_cus_train",

    # Data
    "data_start"            : "2020-05-01",
    "train_predict_split"   : "2020-08-01",

    #"item_cols"         : ["product_type_no", "colour_group_code",
    #                       "perceived_colour_value_id", "perceived_colour_master_id",
    #                       "department_no", "index_code", "index_group_no", "section_no", "garment_group_no"],

    "item_cols"         : ["product_type_no", "colour_group_code",
                           "perceived_colour_value_id", "perceived_colour_master_id",
                           "index_code", "index_group_no", "section_no", "garment_group_no"],

    "cus_cols"          : ["Active", "club_member_status", "fashion_news_frequency", "age"],

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

    # Misc
    "min_ans_size"      : 4


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


def get_item_to_vec_file_name():

    col_hash = hashlib.md5(("".join(args["item_cols"])).encode("utf-8")).hexdigest()

    file_name = "%s_%s_%s.csv" % (args['item_vector_data'],
                                  args['data_start'].replace("-", "_"),
                                  col_hash)

    return file_name


def get_cus_to_vec_file_name():

    col_hash = hashlib.md5(("".join(args["cus_cols"])).encode("utf-8")).hexdigest()

    file_name = "%s_%s.csv" % (args['cus_vector_data'], col_hash)

    return file_name

def gen_items_to_vec_files():

    article_file_name = get_item_to_vec_file_name()

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


def gen_cus_to_vec_files():

    cus_item_file = get_cus_to_vec_file_name()

    if os.path.exists(cus_item_file):
        print("Item Customer file already exist.")
        return
    else:
        print("Item Customer file not exist, generate")

    cus_df = pd.read_csv(args["customer"])
    cus_df = cus_df[["customer_id"] + args["cus_cols"]]

    # Fill NAs
    cus_df["Active"].fillna(0, inplace=True)
    cus_df["club_member_status"].fillna("UN", inplace=True)
    cus_df["fashion_news_frequency"].fillna("UN", inplace=True)
    cus_df["fashion_news_frequency"].fillna("UN", inplace=True)
    cus_df["age"].fillna(200, inplace=True)

    # bin ages
    cus_df["age"] = pd.cut(cus_df["age"], [25, 35, 40, 60, 100])

    print(cus_df.info())
    print(unique_data(cus_df))

    enc = OneHotEncoder()
    enc.fit(cus_df.iloc[:, 1:])
    df = pd.DataFrame(enc.transform(cus_df.iloc[:, 1:]).toarray().astype(np.int32))
    df['customer_id'] = cus_df['customer_id']

    df.to_csv(cus_item_file, index=False)


def gen_item_features(items_df, one_cus_df):

    # create item features
    vec = items_df[items_df['article_id'].isin(one_cus_df)].iloc[:,:-1].sum()

    # norm to sum == 1
    return np.array(vec / vec.sum()).round(decimals=5)




def gen_training_date_set(split_date):

    '''
    if os.path.exists(args["item_cus_train_data"]):
        print("ItemCusNN Train Data already exist.")
        return
    else:
        print("ItemCusNN Train Data not exist, generate")

    # filter useless items
    trans = pd.read_csv(args["transaction"])

    train   = trans[(trans["t_dat"] > args["data_start"]) & (trans["t_dat"] < split_date)]
    predict = trans[trans["t_dat"] > split_date]

    # Only preserve custom appears in both dataset
    train = train[train["customer_id"].isin(predict["customer_id"].unique())]
    '''

    # Temp : For convenience
    #train.to_csv("tmp_train.csv", index=False)
    #predict.to_csv("tmp_predict.csv", index=False)

    train = pd.read_csv("tmp_train.csv") 
    predict = pd.read_csv("tmp_predict.csv")

    # read item & customer file
    item_df = pd.read_csv(get_item_to_vec_file_name())
    customer_df = pd.read_csv(get_cus_to_vec_file_name())

    cid_list = []
    train_list = []
    ans_list = []
    train_len = []
    ans_len = []

    predict_group = predict.groupby(['customer_id'])
    customer_df = customer_df.groupby(['customer_id'])

    #counter = 0
    #batch = 0

    for cid, group in tqdm(train.groupby(["customer_id"])):

        predict_items = predict_group.get_group(cid)['article_id']
        ans = predict_items.unique()

        #if len(ans) < args["min_ans_size"] :
        #    continue

        item_feature = gen_item_features(item_df, group['article_id'])

        all_feature = item_feature.tolist() + customer_df.get_group(cid).iloc[0][:-1].tolist()

        #cid_list.append(cid)

        for a in ans:
            train_list.append(all_feature)
            ans_list.append(a)
            #counter += 1
        #train_len.append(len(group['article_id']))
        #ans_len.append(len(ans))

        #if counter > 50000:

    #### End for

    dic = {
           'train': train_list,
           'answer': ans_list
            }

    pd.DataFrame(data=dic).to_csv(args["item_cus_train_data"] + ".csv", index=False)





def train():
    pass


if __name__ == '__main__':

    # Prepare meta data
    gen_cus_to_vec_files()
    gen_items_to_vec_files()

    gen_training_date_set(args["train_predict_split"])










































