# Factorization Machine models in Flax

This repository provides a Flax implementation of factorization machine models and common datasets in CTR prediction.
The code on this repository was converted from [a pytorch implementation of factorization machine models](https://github.com/rixwew/pytorch-fm) to a flax implementation code.


## Available Datasets

* [MovieLens Dataset](https://grouplens.org/datasets/movielens)


## Available Models
| Model | Reference |
|-------|-----------|
| Logistic Regression | |
| Factorization Machine | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| Field-aware Factorization Machine | [Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) |
| Factorization-Supported Neural Network | [W Zhang, et al. Deep Learning over Multi-field Categorical Data - A Case Study on User Response Prediction, 2016.](https://arxiv.org/abs/1601.02376) |
| Wide&Deep | [HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.](https://arxiv.org/abs/1606.07792) |
| Attentional Factorization Machine | [J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.](https://arxiv.org/abs/1708.04617) |
| Neural Factorization Machine | [X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.](https://arxiv.org/abs/1708.05027) |
| Neural Collaborative Filtering | [X He, et al. Neural Collaborative Filtering, 2017.](https://arxiv.org/abs/1708.05031) |
| Field-aware Neural Factorization Machine | [L Zhang, et al. Field-aware Neural Factorization Machine for Click-Through Rate Prediction, 2019.](https://arxiv.org/abs/1902.09096) |
| DeepFM | [H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247) |



# Reference Code
https://github.com/rixwew/pytorch-fm

## Licence
MIT