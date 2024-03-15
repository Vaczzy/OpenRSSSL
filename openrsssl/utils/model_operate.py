# Copyright (c) Vaczzy
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base_support as bs

def vissl2pth(visslcheckpointFilePath,PthFilePath):
    """
    将VISSL保存的checkpint保存为pth格式
    input: 
        VISSL保存的checkpoint文件路径,格式为.torch
        要保存的pth文件路径
    """
    model=bs.GetVISSLModel(visslcheckpointFilePath)
    bs.SavePthModel(PthFilePath,model)

def visslTrunk2pth(visslcheckpointFilePath,PthFilePath):
    """
    将VISSL保存的checkpint中的trunk保存为pth格式
    input: 
        VISSL保存的checkpoint文件路径,格式为.torch
        要保存的pth文件路径
    """
    trunk=bs.GetModelTrunk(visslcheckpointFilePath)
    bs.SavePthModel(PthFilePath,trunk)

def visslTrunkRename2pth(visslcheckpointFilePath,PthFilePath):
    """
    将VISSL保存的checkpoint中的trunk读取完成后修改层名,并保存为pth格式
    input: 
        visslcheckpointFilePath:VISSL保存的checkpoint文件路径,格式为.torch
        PthFilePath:要保存的pth文件路径
    """
    trunk=bs.ChangeLayerName(visslcheckpointFilePath)
    bs.SavePthModel(PthFilePath,trunk)

def pclTrunkRename2pth(tarcheckpointFilePath,PthFilePath):
    """
    将PCL保存的checkpint中的trunk保存为pth格式
    input: 
        tarcheckpointFilePath:PCL保存的checkpoint文件路径,格式为.pth.tar
        PthFilePath:要保存的pth文件路径
    """
    trunk=bs.ChangePCLlayerName(tarcheckpointFilePath)
    bs.SavePthModel(PthFilePath,trunk)

def mmTrunkRename2pth(mmcheckpointFilePath,PthFilePath):
    """
    使mmselfsup的模型适配mmsegmentation
    """
    # trunk=bs.ChangeMMlayerName(mmcheckpointFilePath) # 需要针对模型进行修改
    print('[zzy note]You must use this code to change BYOL!!!!')
    trunk=bs.ChangeBYOLMMlayerName(mmcheckpointFilePath)
    # trunk=bs.ChangeMMlayerName(mmcheckpointFilePath)
    bs.SavePthModel(PthFilePath,trunk)


def visslViTTrunkRename2pth(visslViTcheckpointFilePath,PthFilePath):
    """
    将VISSL保存的ViT checkpoint中的trunk读取完成后修改层名,并保存为pth格式
    input: 
        visslcheckpointFilePath:VISSL保存的checkpoint文件路径,格式为.torch
        PthFilePath:要保存的pth文件路径
    """
    trunk=bs.ChangeViTLayerName(visslViTcheckpointFilePath)
    bs.SavePthModel(PthFilePath,trunk)

def visslViTTrunkRename2pth_v2(visslViTcheckpointFilePath,PthFilePath):
    """
    将VISSL保存的ViT checkpoint中的trunk读取完成后修改层名,并保存为pth格式
    input: 
        visslcheckpointFilePath:VISSL保存的checkpoint文件路径,格式为.torch
        PthFilePath:要保存的pth文件路径

    修复mmsegmentation无法的bug
    """
    trunk=bs.ChangeViTLayerName_v2(visslViTcheckpointFilePath)
    bs.SavePthModel(PthFilePath,trunk)


def mmViTTrunkRename2pth(mmViTcheckpointFilePath,PthFilePath):
    """
    将mmselfsup保存的ViT checkpoint中的trunk读取完成后修改层名,并保存为pth格式
    input: 
        visslcheckpointFilePath:VISSL保存的checkpoint文件路径,格式为.torch
        PthFilePath:要保存的pth文件路径
    """
    trunk=bs.ChangeMMViTLayerName(mmViTcheckpointFilePath)
    bs.SavePthModel(PthFilePath,trunk)

def ContrastiveCrop2pth(contrastiveCropFilePath,PthFilePath,flag='moco_state'):
    """
    将contrastive crop保存的 checkpoint转为mmsegmentation格式
    input: 
        visslcheckpointFilePath:VISSL保存的checkpoint文件路径,格式为.torch
        PthFilePath:要保存的pth文件路径
    
    """
    trunk=bs.ChangeContrastiveCroplayerName(contrastiveCropFilePath,flag)
    bs.SavePthModel(PthFilePath,trunk)
