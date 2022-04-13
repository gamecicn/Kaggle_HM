import pandas as pd
from tqdm import tqdm
import os
import argparse
import numpy as np
# Utility
import time

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

    # args.data = "D://workspace//Kaggle//HM_Recommend//Data"

    STATE_LEN = args.state_len
    DATA = args.data

    # Read all transaction data
    print('\nStart reading all transaction data ...')
    start_t = time.time()
    raw_data = pd.read_csv(f"{DATA}//transactions_train.csv")
    print(f'Finish reading in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_t))}')

    customer_ids = raw_data['customer_id'].unique()
    customer_size = len(customer_ids)

    item_id      = raw_data['article_id'].unique()
    article_size = len(item_id)

    # convert original id to paper style id
    code_to_item = {value: index for index, value in enumerate(item_id)}
    raw_data['article_id'] = raw_data['article_id'].apply(lambda x: code_to_item[x])


    # Generate data_statis.df for paper format
    if "paper" == args.format:
        dic = {'state_size': [STATE_LEN], 'item_num': [len(raw_data['article_id'].unique())]}
        data_statis = pd.DataFrame(data=dic)
        data_statis.to_pickle(os.path.join(DATA, './data_statis.df'))
        PAD = article_size
    else:
        PAD = 0

    # Sample customer_id for processing
    if args.size == -1:
        target_cus_size = customer_size
    else:
        target_cus_size = args.size

    if args.random:
        np.random.seed(args.seed)
        target_id = np.random.choice(customer_ids, size=target_cus_size, replace=False)
    else:
        target_id = customer_ids[args.cus_start : target_cus_size]


    # Filter and ave sampled_session.df/csv
    print('\nFilter and save all valid sampled data')
    sampled_sessions = raw_data[raw_data['customer_id'].isin(target_id)]
    # only keep sessions with length >= 3
    sampled_sessions.loc[:, 'valid_session'] = sampled_sessions.customer_id.map(sampled_sessions.groupby('customer_id')['article_id'].size() > 2)
    sampled_sessions = sampled_sessions.loc[sampled_sessions.valid_session].drop('valid_session', axis=1)
    
    sampled_sessions.to_csv(os.path.join(DATA, './sampled_sessions.csv'))
    to_pickled_df(DATA, sampled_sessions=sampled_sessions)


    # Count popularities of items and save % to pop_dict.txt
    print('\nStart counting popularity ...')
    start_t = time.time()
    total_actions = sampled_sessions.shape[0]
    pop_dict = {}
    for idx, row in tqdm(sampled_sessions.iterrows()):
        action = row['article_id']
        pop_dict[action] = pop_dict[action] + 1 if action in pop_dict else 1
    
    for action in pop_dict.keys():
        pop_dict[action] = float(pop_dict[action])/float(total_actions)
    
    with open(os.path.join(DATA, 'pop_dict.txt'), 'w') as f:
        f.write(str(pop_dict))
    print(f'Popularity finished in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_t))}')


    # Split into train, val, test
    print('\nStart spliting into train, val, test data ...')
    train_end_id = int(len(target_id) * 0.7)
    val_end_id   = int(len(target_id) * 0.9)

    train_id = target_id[:  train_end_id]
    val_id   = target_id[train_end_id : val_end_id]
    test_id  = target_id[val_end_id: ]
    
    # Save sampled_train, val, test .df
    train_sessions = sampled_sessions[sampled_sessions['customer_id'].isin(train_id)]
    val_sessions = sampled_sessions[sampled_sessions['customer_id'].isin(val_id)]
    test_sessions = sampled_sessions[sampled_sessions['customer_id'].isin(test_id)]
    
    to_pickled_df(DATA, sampled_train=train_sessions)
    to_pickled_df(DATA, sampled_val=val_sessions)
    to_pickled_df(DATA,sampled_test=test_sessions)

    print(f'''
           Generate Replay Buffer:
                Total Customer ID : {target_cus_size}
                     Train:      {len(train_id)} ids | {len(train_sessions)} actions
                     Validation: {len(val_id)} ids | {len(val_sessions)} actions
                     Test:       {len(test_id)} ids | {len(test_sessions)} actions
                     
                Random : {args.random}
                Random Seed : {args.seed}
                Format : {args.format}
    
                Total customer id number : {customer_size}
                Total article id number  : {article_size}
    ''')

    # Generating replay buffer from training data
    print(f"Generating training replay buffer")

    state, len_state, action, is_buy, next_state, len_next_state, is_done, price, channel = [], [], [], [], [], [], [], [], []

    groups = sampled_sessions[sampled_sessions["customer_id"].isin(train_id)].groupby("customer_id")

    for tid, group in tqdm(groups):

        # Skip short history interaction
        # if group.shape[1] < 3:
        #     continue

        history = []
        for index, row in group.iterrows():
            s = list(history)
            len_state.append(STATE_LEN if len(s) >= STATE_LEN else 1 if len(s) == 0 else len(s))
            s = pad_history(s, STATE_LEN, PAD)
            a = row['article_id']
            price.append(row['price'])
            channel.append(row['sales_channel_id'])
            state.append(s)
            action.append(a)
            is_buy.append(1)  # is_buy always 1
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
        reply_buffer.to_pickle(os.path.join(DATA, f'./replay_buffer.df'))
    else:
        reply_buffer.to_csv(os.path.join(DATA, f"./replay_buffer.csv"), index=False)
