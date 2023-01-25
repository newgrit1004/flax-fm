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
| xDeepFM | [J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.](https://arxiv.org/abs/1803.05170) |


# 주피터 노트북 파일 설명

- no_batch_norm_dropout_exists_training.ipynb
    - batch_normalization 을 하지 않고, dropout을 해야하는 모델에 대해서 training할 때 사용하는 주피터 노트북입니다.
    - 모델
        - AttentionalFactorizationMachineModelFlax

- batch_norm_dropout_exist_training.ipynb
    - batch_normalization과 dropout을 포함한 모델에 대해서 training할 때 사용하는 주피터 노트북입니다.
    - 모델
        - WideAndDeepModelFlax
        - FactorizationSupportedNeuralNetworkModelFlax
        - NeuralFactorizationMachineModelFlax
        - NeuralCollaborativeFilteringFlax
        - FieldAwareNeuralFactorizationMachineModelFlax
        - DeepFactorizationMachineModelFlax
        - ExtremeDeepFactorizationMachineModelFlax


- no_batch_norm_dropout_training.ipynb
    - batch_normaliation과 dropout이 포함되지 않은 모델에 대해서 training할 때 사용하는 주피터 노트북입니다.
    - 모델
        - LogisticRegressionModelFlax
        - FactorizationMachineModelFlax
        - FieldAwareFactorizationMachineModelFlax


- compare_pytorch_flax_train_speed.ipynb
    - 동일한 데이터셋(MovieLens20MDataset)에 대해 pytorch로 구현한 FactorizationMachineModel과 Flax로 구현한 FactorizationMachineModel에 대해 각각 모델 트레이닝을 하고 트레이닝 속도 및 loss function 값의 수렴도를 비교한 주피터 노트북 파일입니다.

- compare_pytorch_flax_model_architecture.ipynb
    - 동일한 데이터셋(MovieLens20MDataset)에 대해 pytorch로 구현한 FactorizationMachineModel과 Flax로 구현한 FactorizationMachineModel를  각각 onnx 파일과 tflite 파일로 export하고,
    [netron 라이브러리](https://github.com/lutzroeder/netron)를 이용하여 모델 구조를 시각화하여 비교해봅니다.

# profile_results
- [scalene](https://github.com/plasma-umass/scalene)(a high-performance CPU, GPU and memory profiler for Python)를 사용하여 모델에 대해 profiling한 결과들을 html파일 형태로 업로드하였습니다.
- [htmlviewer](https://codebeautify.org/htmlviewer)을 통해 profile_results로 나온 결과를 볼 수 있습니다.

# TODO
- 모델별 특징과 상관없이 통일된 training 코드 작성
- Inference 코드 작성
- html 파일을 더 쉽게 볼 수 있는 법에 대해 고민해보기
- Dockerfile 설치 과정 효율화
    - 현재는 설치해야하는 용량이 매우 크고 오래 걸리는 편

# Reference Code
https://github.com/rixwew/pytorch-fm

## Licence
MIT