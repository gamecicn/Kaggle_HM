# Reinforcement Learning Regularized Recsys for Kaggle H&M Competition

![HM](./img/title.png)

## Introduction

Xinxin and his team proposed an innovation mechanism to apply Reinforcement Learning as a regularization of supervised learning models for recommender  systems [1].  In the newer paper[2], they advanced introduced negative training samples which can eliminate positive bias in the first paper. Hence  the new models could achieve better performance. The paper's code managed to get State-Of-The-Art performance on RC15[3] and Retailrocket[4] datasets, which are dedicated for recommender system models. However, the recommender system compromise many application environments. We wonder about the generalizability of the method. In this project, we try to apply the second paper's code to data of the latest Kaggle's competition, [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) .  Moreover, we implement other algorithms on the H&M dataset.   Hence, this data set includes our works of two phases: 

- Apply code of *Supervised Advantage Actor-Critic for Recommender Systems* on data from Kaggle H&M competition.
- Build other models for  Kaggle H&M competition.


##  Phase 1: Apply Code on H&M Data

#### Data Preprocessing

Even though the dataset of H&M competition is also prepared for recommender system, it is quite different from RC15 and  Retailrocket which are used in the papers.  Beyond transaction log,  H&M dataset includes items and customer demographic information. Also, there are some major variances between H&M and RC15/Retailrocket's  transaction log: 

- **Size** :   The H&M transaction is much larger than RC15/Retailrocket's. 
  -  RC15                 45 MB
  -  Retailrocket     92 MB
  -  H&M                  3.4 GB
-  **Click & Purchase** :  The RC15/Retailrocket transaction  include 'isbuy' feature which could indicate whether the customer bought the item in this click action. However, the H&M dataset only recorded purchase actions.  This difference brings some difficulties  because the papes' code has some optimization for purchase. 
-  **Timestamp**:   RC15/Retailrocket dataset's timestamp is accurate to second, while H&M's records only have date information. 
- **Session**:  HM data has no session information, so we treat one customer's transactions as one session. While most RC15/Retailrocket sessions have fewer than 50 items, the sessions in the H&M dataset typically have many more items, some as high as 200. Long sessions cause RNN models to require much more memory and GPU times. 

To overcome the above problems, we modified the [data sampling code](https://github.com/gamecicn/Kaggle_HM/blob/main/src/models/gen_replay_buffer.py) to adapt H&M's transaction log to the paper's training code. We only sample training data from 2019 and beyond to ensure the validity of items. Besides, sessions with lengths less than three are removed, and the ones with lengths over 50 limit the evaluation memory usage. Finally, we also commented on the part of the calculation of clicks to avoid the calculation error that the number of clicks is 0.


####  Result 
We updated the training code in Xinxin's paper to meet our experimental needs. Ex. [SA2C Network Code](https://github.com/gamecicn/Kaggle_HM/blob/main/src/models/SA2C.py), [SNQN Network Code](https://github.com/gamecicn/Kaggle_HM/blob/main/src/models/SNQN.py). </br >
Then, we created a series of notebooks to set up a suitable environment on [Colab](https://research.google.com/colaboratory/) and used Colab GPU resources to train remotely. Ex. [SNQN Training Noetbook](https://github.com/gamecicn/Kaggle_HM/blob/main/notebook/HM_SNQN_SASRec.ipynb)


| **Models**  | **HR@5** | **NG@5** | **HR@10** | **NG@10** | **NR@20** | **NG@20** |
| :---------: | :------: | :------: | :-------: | :-------: | :-------: | :-------: |
|  GRU-SNQN   |  0.0074  |  0.0051  |  0.0115   |  0.0065   |  0.0166   |  0.0077   |
|  GRU-SA2C   |  0.0091  |  0.0063  |  0.0129   |  0.0075   |  0.0182   |  0.0089   |
| Caser-SNQN  |  0.0068  |  0.0046  |  0.0101   |  0.0056   |  0.0151   |  0.0069   |
| Caser-SA2C  |  0.0082  |  0.0058  |  0.0111   |  0.0068   |  0.0157   |  0.0079   |
| NItNet-SNQN |  0.0151  |  0.0104  |  0.0216   |  0.0125   |  0.0304   |  0.0147   |
| NItNet-SA2C |  0.0319  |  0.0222  |  0.0410   |  0.0252   |  0.0510   |  0.0277   |
| SASRec-SNQN |  0.0262  |  0.0175  |  0.0381   |  0.0213   |  0.0504   |  0.0244   |
| SASRec-SA2C |  0.0399  |  0.0279  |  0.0530   |  0.0322   |  0.0637   |  0.0349   |


## Phase 2: Other Models

#### Collaborative Filtering 

We implement collaborative filtering in [Notebook:  Collaborative Filtering](https://github.com/gamecicn/Kaggle_HM/blob/main/notebook/Collaborative%20Filtering.ipynb) . This method achieves a Kaggle score of 0.0127 [5].

#### Content Based 

We also build content based models which can use  items and customer demographic information. The source code in  [gen_item_cus_dataset.py](https://github.com/gamecicn/Kaggle_HM/blob/main/src/preprocessing/gen_item_cus_dataset.py) could convert the raw data into supervised learning model friendly structure data.  In [ContentBase_NN](https://github.com/gamecicn/Kaggle_HM/blob/main/notebook/ContentBase_NN.ipynb) , we use PyTorch to implement  a Deep Neural Network to generate a recommendation. However, we fail to train the full model because it triggers out of memory issues on Google Colab Pro. 
 

## Code Structure
```
├── README.md
├── img
│   └── title.png
├── notebook
│   ├── Collaborative Filtering.ipynb
│   ├── EDA.ipynb
│   ├── GRU.ipynb
│   └── HM_SNQN_SASRec.ipynb
├── paper
│   ├── 2006.05779.pdf
│   └── 2111.03474.pdf
└── src
    ├── models
    │   ├── GRU.py
    │   ├── NextItNetModules.py
    │   ├── SA2C.py
    │   ├── SASRecModules.py
    │   ├── SNQN.py
    │   ├── customer_self_freq.py
    │   ├── gen_replay_buffer.py
    │   └── utility.py
    └── preprocessing
        ├── content_vec_simi.py
        ├── gen_item_cus_dataset.py
        └── gen_sample_ref.py
```

## Reference

[1] Xin, Xin, et al. "Self-supervised reinforcement learning for recommender systems." Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020.

[2] Xin, Xin, et al. "Supervised Advantage Actor-Critic for Recommender Systems." Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 2022.

[3]  [RecSys 2015 – Challenge – RecSys (acm.org)](https://recsys.acm.org/recsys15/challenge/)

[4] [Retailrocket recommender system dataset | Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

[5] [Evaluation of H&M Competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview/evaluation)


















