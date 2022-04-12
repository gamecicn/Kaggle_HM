
import numpy as np
import pandas as pd
from tqdm import tqdm

#TRANS_DATA = "../data/trans_500.csv"

TRANS_DATA = "../../Data/transactions_train.csv"
SAMPLE_SUBMIT = "../data/sample_submission.csv"

trans_data = pd.read_csv(TRANS_DATA)
sample = pd.read_csv(SAMPLE_SUBMIT)

groups = trans_data.groupby(['customer_id'])

customer_id = []
articel = []
price = []
channel = []

for cid in tqdm(sample["customer_id"]):

    customer_id.append(cid)

    if cid in groups.indices.keys():
        cdf = groups.get_group(cid)
        articel.append(cdf["article_id"].tolist())
        #price.append(cdf["price"])
        channel.append(cdf["sales_channel_id"].tolist())
    else:
        articel.append([])
        #price.append([])
        channel.append([])

####
out_data = {
    "customer_id": customer_id,
    'article_id' : articel,
    'channel' : channel
}

df = pd.DataFrame(data=out_data)

df.to_csv("submit_ref.csv", index=False)





