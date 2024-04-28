# Installation

This repository is built in PyTorch 1.11 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

## 1. Clone our repository
```shell
git clone git@github.com:hollinsStuart/dark.git
cd dark
```

## 2. Make conda environment
```shell
conda create -n dark python=3.7
conda activate dark
```

## 3. Install dependencies
### CUDA Toolkit
**You will need to install the correct CUDA toolkit of your version.** Use the following command to check.
```shell
nvidia-smi
```
Please visit https://developer.nvidia.com/cuda-downloads for downloading the CUDA Toolkit of your version. Chances are that you will need to uninstall the current cuda toolkit and reinstall again if they do not match in the first place.

Use 
```nvcc --version``` to check whether your cuda toolkit is successfully installed on your computer.

### Packages
```shell
pip install -r requirements.txt
```

## 4. Install basicsr
```shell
python setup.py develop
```

## 5. Download Dataset:
We use the following datasets:
Lol_train  https://drive.google.com/file/d/1K29vsPfMUsAkYvmNLcaUgiOEYGMxFydd/view?usp=sharing
Lol_test  https://drive.google.com/file/d/1jUGpsih3T-1H7t3gqpEdj7ZD5GcU_v0m/view?usp=sharing

## 6. Modify the configuration file
Please modify the parameters in Enhancement/Options/dark_train_config.yml

## 7. Train the model
```shell
python3 basicsr/train.py -opt Enhancement/Options/dark_train_config.yml
```