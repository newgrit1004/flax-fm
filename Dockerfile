FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt update && apt install python3-pip jupyter-notebook -y
WORKDIR /dist

RUN pip install --upgrade setuptools pip
RUN pip install nvidia-pyindex
RUN pip install nvidia-tensorrt==8.0.0.3
RUN pip install tensorflow
RUN pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install flax==0.6.2
RUN pip install -U jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install torch==1.13.0
RUN pip3 install pandas==1.5.1