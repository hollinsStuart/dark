# Installation

This repository is built in PyTorch 1.11 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

1. Clone our repository
```shell
git clone https://github.com/swz30/MIRNetv2.git
cd MIRNetv2
```

2. Make conda environment
```shell
conda create -n torchMIRNet python=3.7
conda activate torchMIRNet
```

3. Install dependencies
**Change the cudatoolkit version to YOUR CUDA version.** We are using 12.4 as an example. Use the following command to check your cuda version.
```shell
nvidia-smi
```
```shell
conda install pytorch=1.13 torchvision -c pytorch
pip install -r requirements.txt
```

4. Install basicsr
```shell
python setup.py develop --no_cuda_ext
```

```shell
python3 basicsr/train.py -opt Enhancement/Options/dark_train_config.yml
```