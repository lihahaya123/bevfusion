# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install -y wget build-essential g++ gcc
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda (use py38 version to avoid anaconda-auth conflict with latest Miniconda)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda config --set show_channel_urls yes && \
    conda config --set remote_connect_timeout_secs 60 && \
    conda config --set remote_read_timeout_secs 300 && \
    conda config --set remote_max_retries 5 && \
    (conda config --remove channels defaults || true) && \
    conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --set channel_priority flexible

RUN conda install python=3.8 -y && conda clean -afy

# Use Tsinghua mirror for faster download in China
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10

RUN --mount=type=cache,target=/root/.cache/torch-wheels \
    wget --continue --tries=0 --timeout=60 --read-timeout=300 \
        -O /root/.cache/torch-wheels/torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu113/torch-1.10.1%2Bcu113-cp38-cp38-linux_x86_64.whl && \
    wget --continue --tries=0 --timeout=60 --read-timeout=300 \
        -O /root/.cache/torch-wheels/torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu113/torchvision-0.11.2%2Bcu113-cp38-cp38-linux_x86_64.whl && \
    pip install --no-cache-dir \
        /root/.cache/torch-wheels/torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl \
        /root/.cache/torch-wheels/torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl \
        torchaudio==0.10.1

RUN pip install Pillow==8.4.0 \
    numpy==1.23.5 \
    tqdm \
    torchpack \
    mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0 \
    nuscenes-devkit \
    mpi4py==3.0.3 \
    numba==0.48.0

# Replace opencv-python with headless version (avoids Qt dependency issues)
RUN pip uninstall opencv-python -y && pip install opencv-python-headless

# Fix compatibility issues with yapf and setuptools for mmcv 1.4.0
RUN pip install yapf==0.40.1 setuptools==59.5.0
