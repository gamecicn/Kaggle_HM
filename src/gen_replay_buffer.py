import pandas as pd
from tqdm import tqdm
import os
import argparse
import numpy as np
# Utility

def parse_args():

    parser = argparse.ArgumentParser(description="Generae replay buffer data.")

    parser.add_argument('--data', nargs='?', default='data',
                        help='data directory')

    parser.add_argument('--state_len', type=int, default=10,
                        help='Max state length.')

    parser.add_argument('--size', type=int, default=-1,
                        help='How many customer id will be used to generate train/val/test id.')

    parser.add_argument('--random', type=bool, default=True,
                        help='Is select custom id randomly. If True, then cus_start will be invalided')

    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')

    parser.add_argument('--cus_start', type=int, default=0,
                        help='Generate from the nth customer id')

    parser.add_argument('--format', choices=['paper', 'csv'], default='paper',
                        help='Output format "paper" (paper format) or "csv" (csv file)')

    #parser.add_argument('--pad', choices=['item_size', '0'], default='paper',
    #                    help='Use which mark ("item_size", "0") as pad')

    #parser.add_argument('--train_split', type=float, default='0.8',
    #                    help='Split into training and testing data')

    return parser.parse_args()


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))


def pad_history(itemlist, length, pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


###########################################################

# Usage
'''
python gen_replay_buffer.py  --size  20000  --format  paper
python gen_replay_buffer.py  --size  20000  --format  csv
'''

if __name__ == '__main__':

    args = parse_args()

    args.data = "D://workspace//Kaggle//HM_Recommend//Data"

    STATE_LEN = args.state_len
    DATA = args.data

    # read
    raw = pd.read_csv(f"{DATA}//transactions_train.csv")

    customer_ids = raw['customer_id'].unique()
    customer_size = len(customer_ids)


    item_id      = raw['article_id'].unique()
    article_size = len(item_id)

    # convert original id to paper style id
    code_to_item = {value: index for index, value in enumerate(item_id)}
    raw['article_id'] = raw['article_id'].apply(lambda x: code_to_item[x])


    # Generate data_statis.df for paper format
    if "paper" == args.format:
        dic = {'state_size': [STATE_LEN], 'item_num': [len(raw['article_id'].unique())]}
        data_statis = pd.DataFrame(data=dic)
        data_statis.to_pickle(os.path.join(f'./data_statis.df'))
        PAD = article_size
    else:
        PAD = 0

    # Calculate id to process
    if args.size == -1:
        target_cus_size = customer_size
    else:
        target_cus_size = args.size

    if args.random:
        np.random.seed(args.seed)
        target_id = np.random.choice(customer_ids, size=target_cus_size, replace=False)
    else:
        target_id = customer_ids[args.cus_start : target_cus_size]

    train_end_id = int(len(target_id) * 0.7)
    val_end_id   = int(len(target_id) * 0.9)

    train_id = target_id[:  train_end_id]
    val_id   = target_id[train_end_id : val_end_id]
    test_id  = target_id[val_end_id: ]

    print(f'''
           Generate Replay Buffer:
                Total Customer ID : {target_cus_size}
                     Train:      {len(train_id)}
                     Validation: {len(val_id)}
                     Test:       {len(test_id)}
                     
                Random : {args.random}
                Random Seed : {args.seed}
                Format : {args.format}
    
                Total customer id number : {customer_size}
                Total article id number  : {article_size}
    ''')

    for seg_id, sub_target_id in enumerate([train_id, val_id, test_id]):

        data_set_name = ["train", "val", "test"][seg_id]

        print(f"Genearting {data_set_name}")

        state, len_state, action, is_buy, next_state, len_next_state, is_done, price, channel = [], [], [], [], [], [], [], [], []

        groups = raw[raw["customer_id"].isin(sub_target_id)].groupby("customer_id")

        for tid, gorup in tqdm(groups):

            # Skip short history interaction
            if gorup.shape[1] < 3:
                continue

            history = []
            for index, row in gorup.iterrows():
                s = list(history)
                len_state.append(STATE_LEN if len(s) >= STATE_LEN else 1 if len(s) == 0 else len(s))
                s = pad_history(s, STATE_LEN, PAD)
                a = row['article_id']
                price.append(row['price'])
                channel.append(row['sales_channel_id'])
                state.append(s)
                action.append(a)
                is_buy.append(0)  # is_buy always 0
                history.append(row['article_id'])
                next_s = list(history)
                len_next_state.append(STATE_LEN if len(next_s) >= STATE_LEN else 1 if len(next_s) == 0 else len(next_s))
                next_s = pad_history(next_s, STATE_LEN, PAD)
                next_state.append(next_s)
                is_done.append(False)
            is_done[-1] = True

        dic={'state':state,
             'len_state':len_state,
             'action':action,
             'is_buy':is_buy,
             'next_state':next_state,
             'len_next_states':len_next_state,
             'price' : price,
             'channel' : channel,
             'is_done':is_done}

        reply_buffer=pd.DataFrame(data=dic)

        if "paper" == args.format:
            reply_buffer.to_pickle(os.path.join(f'./reply_buffer_{data_set_name}.df'))
        else:
            reply_buffer.to_csv(f"./reply_buffer_{data_set_name}.csv", index=False)

























