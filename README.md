# Reinforcement Learning Regulation Recsys for Kaggle H&M Competition

![HM](./img/title.png)

## Introduction

Xinxin and his team proposed an innovation mechanism to apply Reinforcement Learning as a regulation on supervised learning models for recommend systems [1].  In the newer paper[2], they advanced introduced negative training samples which can eliminate positive bias in the first paper, hence the new models could achieve better performance. The paper's code managed to get State-Of-The-Art performance on RC15[3] and Retailrocket[4] dataset, which are dedicate for recommend system models. However, the recommend system compromise many application environment. We wonder the generalizability of the method. In this project we try to apply the second paper's code on data of a latest Kaggle's competition, [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) .  Moreover, we implement other algorithms on H&M dataset.   Hence, this data set include our works of two phases: 

- Apply code of *Supervised Advantage Actor-Critic for Recommender Systems* on data from Kaggle H&M competition.
- Build other models for  Kaggle H&M competition.


##  Phase 1: Apply Code on H&M Data

#### Data Preprocessing

Even though the dataset of H&M competition is also prepared for recommend system, it is quite different from RC15 and  Retailrocket which are used in the papers.  Beyond transaction log,  H&M dataset includes items and customer demographic information. Also, there are some major variances between H&M and RC15/Retailrocket's  transaction log: 

- **Size** :   The H&M transaction is much larger. 
  -  RC15                 45 MB
  -  Retailrocket     92 MB
  -  H&M                  3.4 GB
-  **Click & Purchase** :  The RC15/Retailrocket transaction  include 'isbuy' feature which could indicate whether customer bought the item in this click action. However, H&M dataset only recorded purchase actions.  This different brings some difficult because the papes's code has some optimization for purchase. 
-  **Timestamp**:   RC15/Retailrocket dataset's timestamp accurate to second, while H&M's records only has date information. 
- **Session**:  RC15/Retailrocket sessions are no more than 50 items, while there are some sessions in H&M dataset are longer than 200 items. The long session would cause RNN model require much more memory and GPU times. 

To overcome above problems, we transformed H&M's transaction log to fit the paper's code. 



####  Result 

| **Models**  | **HR@5** | **NG@5** | **HR@10** | **NG@10** | **NR@20** | **NG@20** |
| :---------: | :------: | :------: | :-------: | :-------: | :-------: | :-------: |
|  GRU-SNQN   |  0.0074  |  0.0051  |  0.0115   |  0.0065   |  0.0166   |  0.0077   |
|  GRU-SA2C   |  0.0091  |  0.0063  |  0.0129   |  0.0075   |  0.0182   |  0.0089   |
| Caser-SNQN  |  0.0068  |  0.0046  |  0.0101   |  0.0056   |  0.0151   |  0.0069   |
| Caser-SA2C  |  0.0082  |  0.0058  |  0.0111   |  0.0068   |  0.0157   |  0.0079   |
| NItNet-SNQN |  0.0151  |  0.0104  |  0.0216   |  0.0125   |  0.0304   |  0.0147   |
| NItNet-SA2C |  0.6080  |  0.5284  |  0.6522   |  0.5430   |  0.6853   |  0.5514   |
| SASRec-SNQN |  0.0262  |  0.0175  |  0.0381   |  0.0213   |  0.0504   |  0.0244   |
| SASRec-SA2C |  0.0399  |  0.0279  |  0.0530   |  0.0322   |  0.0637   |  0.0349   |

####  Conclusion



## Phase 2: Other Methods






## Reference

[1] Xin, Xin, et al. "Self-supervised reinforcement learning for recommender systems." Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020.

[2] Xin, Xin, et al. "Supervised Advantage Actor-Critic for Recommender Systems." Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 2022.

[3]  [RecSys 2015 – Challenge – RecSys (acm.org)](https://recsys.acm.org/recsys15/challenge/)

[4] [Retailrocket recommender system dataset | Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)



















