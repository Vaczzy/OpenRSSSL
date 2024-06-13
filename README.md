<p align="center">
    <img src="docs/OpenRSSSL.png" width="500" />
</p>

<p align="center">
    <a href="https://pytorch.org/get-started/previous-versions/"><img src="https://img.shields.io/badge/python-3.8~3.9-red"></a>
    <a href="https://pytorch.org/get-started/previous-versions/"><img src="https://img.shields.io/badge/pytorch-2.0-blue"></a>
    <a href="https://developer.nvidia.com/cuda-downloads"><img src="https://img.shields.io/badge/cuda-11.7~11.8-orange"></a>
    <a href="https://github.com/facebookresearch/vissl"><img src="https://img.shields.io/badge/vissl-0.1.5-yellow"></a>
    <a href="https://github.com/open-mmlab/mmsegmentation"><img src="https://img.shields.io/badge/mmseg-red"></a>
    <a href="https://img.shields.io/github/license/Vaczzy/OpenRSSSL"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

<p align="center">
  <a href="#TODO List">TODO List</a> •
  <a href="##Version Record">Version Record</a> •
  <a href="#Installation">Installation</a> •  
  <a href="#Acknowledgement">Acknowledgement</a> •
  <a href="#Citation">Citation</a>
</p>
<br>
<p align="center">
    Open Source Remote Sensing Self-Supervised Learning.
    The repository is OPEN, Pull requests are welcome!
</p>

## TODO List

- [x] Create Stable VISSL Version
- [x] Simplify Installation Process
- [x] Reduced Dependency Package
- [x] Add Actions
- [x] Complete CI/CD
- [x] Simplify Pretrain Config File [Add PROC_ID]
- [ ] Complete the Pretrain Process
- [ ] Check the VISSL Pretrain
- [ ] Add Remote Sensing Image Segmentation Code
- [ ] Add Remote Sensing Image Classfication Code
- [ ] Add Remote Sensing Image Object Detection Code
- [ ] Organize the Code with LangChain Style
- [ ] Create Web Page
- [ ] Create stable OpenRSSSL Version
- [ ] Complete the Whole Process: From Pretrain to Specific-Task
- [ ] Make Stable Fairscale

## Version Record
2024-03-03 openrsssl (alpha version)

## Installation

1. Create Environment and Install [Pytorch](https://pytorch.org/)
```
conda create -n openrsssl_env python=3.8 -y
conda activate openrsssl_env
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```
2. Install OpenRSSSL

Download OpenRSSSL source code and switch to the source path for installation:
```
pip install --progress-bar off -r requirements.txt
pip install -e .[dev]
```
3. Optional: Install Apex from source
```
git clone --recursive https://www.github.com/NVIDIA/apex
cd apex
python3 setup.py install
```

## Support Self-supervised Method

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Supported Backbones</b>
      </td>
      <td>
        <b>General Self-supervised Learning Method</b>
      </td>
      <td>
        <b>Remote Sensing Self-supervised Learning Method</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li>ResNet</li>
        <li>ResNeXt</li>
        <li>RegNet</li>
        <li>MobileNet</li>
        <li>Swin-Transformer</li>
        <li>EfficientNet</li>
        <li>ConvNeXt</li>
        <li>BEiT</li>
        <li>XCiT</li>
        </ul>
      </td>
      <td>
        <ul>
        Contrastive:
        <li>SimCLR</li>
        <li>MoCo</li>
        <li>Barlow Twins</li>
        <li>SwAV</li>
        Generative:
        <li>MSN</li>
        Self-Distillation:
        <li>DINO (DeiT)</li>
        <li>iBOT (DeiT)</li>
        Coming soon:
        <li>BYOL *</li>
        <li>DenseCL *</li>
        <li>SimSiam *</li>
        <li>MAE *</li>
        <li>SimMIM *</li>
        <li>EVA *</li>
        <li>Deconstructing Denoising Diffusion Models for Self-Supervised Learning *</li>
        <li>DINO v2</li>
        <li>Jigsaw</li>
        </ul>
      </td>
      <td>
        <ul>
        Coming soon:
        <li>GLCNet (TGRS) *</li>
        <li>FALSE (GRSL) (coming soon.....)</li>
        <li>GraSS (TGRS) (coming soon.....)</li>
        </ul>
      </td>
  </tbody>
</table>

## Acknowledgement
We would like to thank the [VISSL](https://github.com/facebookresearch/vissl) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for its open-source project.