# Environment Debug....

# Installation & Data Preparetion

## Requirements

* Linux
* Python>=3.6.2 and < 3.9
* PyTorch>=1.4
* torchvision (matching PyTorch install)
* CUDA (must be a version supported by the pytorch version)
* OpenCV
* scikit-image
* importlib-resources=5.12.0

## Installing OpenRSSSL (beta)

1. Create Enviroment
```
conda create -n openrsssl_env python=3.8
conda activate openrsssl_env
```
2. Install PyTorch
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Or Visit [Pytorch](https://pytorch.org/) to install

3. Install Apex(optional)
```
pip install packaging
git clone --recursive https://www.github.com/NVIDIA/apex
cd apex
python3 setup.py install
```

4. Install opencv, scikit-image and importlib-resources
```
pip install opencv-python
pip install scikit-image
pip install importlib-resources==5.12.0
```

5. Install OpenRSSSL

Download OpenRSSSL source code and switch to the source path for installation:

```
git clone --recursive https://github.com/Vaczzy/OpenRSSSL.git
cd OpenRSSSL
pip install --progress-bar off -r requirements.txt
pip install classy-vision@https://github.com/Vaczzy/ClassyVision/tarball/master
pip install -e .[dev]
```

## SSL Pretraining:
```
python tools/run_distributed_engines.py config=pretrain/GraSS/grass_1gpu_resnet_b256.yaml \
config.DATA.TRAIN.DATASET_NAMES=["loveda_urban"]
```