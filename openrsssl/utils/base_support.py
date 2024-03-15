# Copyright (c) Vaczzy
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from copy import deepcopy
from iopath.common.file_io import g_pathmgr

def ReadMMSelfModelMethod(mmCheckpointFilePath,device='cpu'):
    """
    读取mmselfsup训练得到的checkpoint
    """
    print('notion: the device is '+device)
    c1=mmCheckpointFilePath.split('.')[-1]
    if c1=='pth':
        with open(mmCheckpointFilePath, "rb") as f:
            mmcheckpoint = torch.load(f,map_location=device)
    else:
        print('Please Check .pth checkpoint file Name!')
        mmcheckpoint=None
    return mmcheckpoint

def GetMMSelfModel(tarCheckpointFilePath):
    """
    主要支持mmself保存的checkpoint
    """
    vc=ReadMMSelfModelMethod(tarCheckpointFilePath)
    model=vc['state_dict']
    return model

def ReadTARModelMethod(tarCheckpointFilePath,device='cpu'):
    """
    读取.pth.tar格式的checkpoint文件路径
    """
    print('notion: the device is '+device)
    c1=tarCheckpointFilePath.split('.')[-1]
    c2=tarCheckpointFilePath.split('.')[-2]
    if (c1=='tar')&(c2=='pth'):
        with open(tarCheckpointFilePath, "rb") as f:
            visslcheckpoint = torch.load(f,map_location=device)
    else:
        print('Please Check .pth.tar checkpoint file Name!')
        visslcheckpoint=None
    return visslcheckpoint


def GetModel(tarCheckpointFilePath):
    """
    主要支持pcl保存的checkpoint
    """
    vc=ReadTARModelMethod(tarCheckpointFilePath)
    model=vc['state_dict']
    return model

def ReadContrastiveCropModel(CheckpointFilePath,device='cpu'):
    """
    读取Contrastive Crop得到的模型
    """
    print('notion: the device is '+device)
    c1=CheckpointFilePath.split('.')[-1]
    if c1=='pth':
        with open(CheckpointFilePath, "rb") as f:
            checkpoint = torch.load(f,map_location=device)
    else:
        print('Please Check .pth checkpoint file Name!')
        checkpoint=None
    return checkpoint

def GetContrastiveCropModel(CheckpointFilePath,flag='moco_state'):
    """
    主要支持contrative crop保存的checkpoint
    """
    vc=ReadContrastiveCropModel(CheckpointFilePath)
    model=vc[flag]
    return model

def ReadVISSLModelMethod(visslCheckpointFilePath,device='cpu'):
    """
    读取VISSL模型的基础实现,resource:VISSL
    input: 
        VISSL保存的checkpoint文件路径,格式为.torch
    return visslcheckpoint 
        数据类型:dict
        顶层字段：[phase_idx,iteration,loss,iteration_num,train_phase_idx,classy_state_dict,type]
        classy_state_dict的字段:
        [train, base_model, meters,optimizer,phase_idx,train_phase_idx,num_updates,losses,hooks,loss,train_dataset_iterator]
    """
    print('notion: the device is '+device)
    control=visslCheckpointFilePath.split('.')[-1]
    if control=='torch':
        with g_pathmgr.open(visslCheckpointFilePath, "rb") as f:
            visslcheckpoint = torch.load(f,map_location=device)
    else:
        print('Please Check VISSL checkpoint file Name!')
        visslcheckpoint=None
    return visslcheckpoint
    

def GetVISSLModel(visslCheckpointFilePath):
    """
    获取可包含为pth格式的模型权重数据
    input: 
        VISSL保存的checkpoint文件路径,格式为.torch
        device :可选择
    output:
       可保存为pth的模型权重数据
       顶层字段：['trunk'],['heads']
    """
    vc=ReadVISSLModelMethod(visslCheckpointFilePath)
    model=vc['classy_state_dict']['base_model']['model']
    return model

def GetModelTrunk(visslCheckpointFilePath):
    """
    input:
        VISSL保存的checkpoint文件路径,格式为.torch
    output:
        模型trunk的权重数据
    """
    vc=ReadVISSLModelMethod(visslCheckpointFilePath)
    model_trunk=vc['classy_state_dict']['base_model']['model']['trunk']
    return model_trunk

def GetModelHeads(visslCheckpointFilePath):
    """
    input: 
        VISSL保存的checkpoint文件路径,格式为.torch
    output:
        模型heads的权重数据
    """
    vc=ReadVISSLModelMethod(visslCheckpointFilePath)
    model_heads=vc['classy_state_dict']['base_model']['model']['heads']
    return model_heads

def SavePthModel(pthFilePath,model):
    """
    保存pth格式的权重文件
    """
    control=pthFilePath.split('.')[-1]
    if control=='pth':
        with open(pthFilePath,'wb') as f:
            torch.save(model, f)
    else:
        print('Please Check save file name, should be .pth')

def SaveVISSLModel(VissltorchFilePath,model):
    """
    保存vissl  .torch格式的权重文件
    !!!!注意:与SavePthModel不同,model=ReadVISSLModelMethod()读取的文件格式
    """
    # vissl save checkpoint method
    # outputFileName='test.torch'
    control=VissltorchFilePath.split('.')[-1]
    if control=='torch':
        with g_pathmgr.open(VissltorchFilePath, "wb") as f:
            torch.save(model, f)
    else:
        print('Please Check save file name, should be .torch')

def ChangeLayerName(visslCheckpointFilePath):
    """
    修改VISSL权重文件层名为MMSegmentation可读取的格式 (目前仅测试了ResNet)
    input: 
        VISSL保存的checkpoint文件路径,格式为.torch 
    output:
        模型修改完成层名的trunk
    """
    trunk=GetModelTrunk(visslCheckpointFilePath)
    trunk_new=deepcopy(trunk) # 需要进行深拷贝
    for name,value in trunk.items():
        new_name=name.split('blocks.')[1]
        trunk_new[new_name]=trunk_new.pop(name)
    return trunk_new

def ChangePCLlayerName(tarCheckpointFilePath):
    """
    修改pcl得到的checkpoint名称
    """
    model=GetModel(tarCheckpointFilePath)

    trunk_tmp=deepcopy(model) # 需要进行深拷贝
    # 筛选得到需要的层
    keys=[]
    for name,value in model.items():
        if name.split('encoder_q.')[0]=='module.':
            keys.append(name)
    trunk_tmp = {k:trunk_tmp[k] for k in keys}

    trunk_new=deepcopy(trunk_tmp) # 需要进行深拷贝
    # 修改层名
    for tmp_name,value in trunk_tmp.items():
        new_name=tmp_name.split('encoder_q.')[1]
        trunk_new[new_name]=trunk_new.pop(tmp_name)
    return trunk_new

def ChangeMMlayerName(mmCheckpointFilePath):
    """
    修改mmselfsup得到的checkpoint名称,适用于densecl模型
    """
    model=GetMMSelfModel(mmCheckpointFilePath)

    trunk_tmp=deepcopy(model) # 需要进行深拷贝
    # 筛选得到需要的层
    keys=[]
    for name,value in model.items():
        if 'encoder_q.0' in name:
            keys.append(name)
    trunk_tmp = {k:trunk_tmp[k] for k in keys}

    trunk_new=deepcopy(trunk_tmp) # 需要进行深拷贝
    # 修改层名
    for tmp_name,value in trunk_tmp.items():
        new_name=tmp_name.split('encoder_q.0.')[1]
        trunk_new[new_name]=trunk_new.pop(tmp_name)
    return trunk_new

def ChangeBYOLMMlayerName(mmCheckpointFilePath):
    """
    修改mmselfsup得到的checkpoint名称,适用于byol模型
    """
    model=GetMMSelfModel(mmCheckpointFilePath)

    trunk_tmp=deepcopy(model) # 需要进行深拷贝
    # 筛选得到需要的层
    keys=[]
    for name,value in model.items():
        if 'online_net.0.' in name:
            keys.append(name)
    trunk_tmp = {k:trunk_tmp[k] for k in keys}

    trunk_new=deepcopy(trunk_tmp) # 需要进行深拷贝
    # 修改层名
    for tmp_name,value in trunk_tmp.items():
        new_name=tmp_name.split('online_net.0.')[1]
        trunk_new[new_name]=trunk_new.pop(tmp_name)
    return trunk_new

def ChangeViTLayerName(visslViTcheckpointFilePath):
    """
    修改vissl得到的vision transformer 名称
    
    当前在vissl上训练的模型没有blocks.0.attn.qkv.bias,计划先不对其进行改变
    """
    trunk=GetModelTrunk(visslViTcheckpointFilePath)
    trunk_new=deepcopy(trunk) # 需要进行深拷贝
    
    #---------------------------------------------------------------------------------------------
    """
    MMSegModel Layer Name:                                            VISSLModel Layer Name:
    backbone.cls_token                                                |class_token
    backbone.pos_embed                                                |pos_embedding
    backbone.pos_embed.projection.weight                              |patch_embed.proj.weight
    backbone.pos_embed.projection.bias                                |patch_embed.proj.bias
           
         backbone.layers.0.ln1.weight                                 |blocks.0.norm1.weight
         backbone.layers.0.ln1.bias                                   |blocks.0.norm1.bias
         backbone.layers.0.attn.attn.in_proj_weight                   |blocks.0.attn.qkv.weight
         backbone.layers.0.attn.attn.in_proj_bias # 一些模型为False     |# blocks.0.attn.qkv.bias # 为了方便,在VISSL训练时,此处设定为True
         backbone.layers.0.attn.attn.out_proj.weight                  |blocks.0.attn.proj.weight
    x12  backbone.layers.0.attn.attn.out_proj.bias                    |blocks.0.attn.proj.bias
         backbone.layers.0.ln2.weight                                 |blocks.0.norm2.weight 
         backbone.layers.0.ln2.bias                                   |blocks.0.norm2.bias 
         backbone.layers.0.ffn.layers.0.0.weight                      |blocks.0.mlp.fc1.weight 
         backbone.layers.0.ffn.layers.0.0.bias                        |blocks.0.mlp.fc1.bias 
         backbone.layers.0.ffn.layers.1.weight                        |blocks.0.mlp.fc2.weight
         backbone.layers.0.ffn.layers.1.bias                          |blocks.0.mlp.fc2.bias

    backbone.ln1.weight                                               |norm.weight
    backbone.ln1.bias                                                 |norm.bias
    """
    #----------------------------------------------------------------------------------------
    pre_all='backbone.' # mmsegmentation对backbone的统一前缀
    nameList=['cls_token','pos_embed','pos_embed.projection.','pos_embed.projection.'] # mmsegmentation中vit的patch的特征提取和投影层
    pre_layer='layers.' # mmsegmentation自注意力模块的统一前缀
    nameList_LN=['ln1.','ln2.'] # 归一化层的名称前缀
    nameList_qkv=['attn.attn.in_proj_weight','attn.attn.in_proj_bias','attn.attn.out_proj.'] # 自注意力层的名称前缀
    nameList_MLP=['ffn.layers.0.0.','ffn.layers.1.'] # 自注意力模块统一前缀

    LayerNum=12 # 自注意力模块统一的层数 编号0-11
    NameCount=0 # NameCount从0开始计数
    # 对于vissl的vit的每一层
    for name,value in trunk.items():
        if NameCount<2:
            """
            0:class_token->backbone.cls_token
            1:pos_embedding->backbone.pos_embed
            """
            new_name=pre_all+nameList[NameCount] 
        elif NameCount<4:
            """
            2:patch_embed.proj.weight->backbone.pos_embed.projection.weight
            3:patch_embed.proj.bias->backbone.pos_embed.projection.bias
            """
            new_name=pre_all+nameList[NameCount]+name.split('.')[-1]
        # 自注意力层名称替换
        elif NameCount<(12*LayerNum+4):
            """
            0 4+i*12:blocks.0.norm1.weight->backbone.layers.0.ln1.weight
            1 5:blocks.0.norm1.bias->       backbone.layers.0.ln1.bias
            2 6:blocks.0.attn.qkv.weight->  backbone.layers.0.attn.attn.in_proj_weight
            3 7:blocks.0.attn.qkv.bias->    backbone.layers.0.attn.attn.in_proj_bias     
            4 8:blocks.0.attn.proj.weight-> backbone.layers.0.attn.attn.out_proj.weight
            5 9:blocks.0.attn.proj.bias->   backbone.layers.0.attn.attn.out_proj.bias  
            6 10:blocks.0.norm2.weight->    backbone.layers.0.ln2.weight
            7 11:blocks.0.norm2.bias->      backbone.layers.0.ln2.bias
            8 12:blocks.0.mlp.fc1.weight->  backbone.layers.0.ffn.layers.0.0.weight
            9 13:blocks.0.mlp.fc1.bias->    backbone.layers.0.ffn.layers.0.0.bias 
            10 14:blocks.0.mlp.fc2.weight->  backbone.layers.0.ffn.layers.1.weight
            11 15:blocks.0.mlp.fc2.bias->    backbone.layers.0.ffn.layers.1.bias
            """
            # 第n(从0开始计数)个注意力模块：n=int((NameCount-4)/12)
            n=int((NameCount-4)/12)
            i=NameCount-4-n*12 # 当前注意力模块的第i层
            if i<2:
                a=name.split('.')[-1]
                new_name=pre_all+pre_layer+str(n)+'.'+nameList_LN[0]+name.split('.')[-1]
            elif i<4:
                new_name=pre_all+pre_layer+str(n)+'.'+nameList_qkv[i-2]
            elif i<6:
                new_name=pre_all+pre_layer+str(n)+'.'+nameList_qkv[-1]+name.split('.')[-1]
            elif i<8:
                new_name=pre_all+pre_layer+str(n)+'.'+nameList_LN[1]+name.split('.')[-1]
            elif i<10:
                new_name=pre_all+pre_layer+str(n)+'.'+nameList_MLP[0]+name.split('.')[-1]
            else:
                new_name=pre_all+pre_layer+str(n)+'.'+nameList_MLP[1]+name.split('.')[-1]
        else:
            """
            norm.weight->backbone.ln1.weight
            norm.bias->  backbone.ln1.bias
            """
            new_name=pre_all+nameList_LN[0]+name.split('.')[-1]
        trunk_new[new_name]=trunk_new.pop(name)
        NameCount+=1
    return trunk_new

def ChangeViTLayerName_v2(visslViTcheckpointFilePath):
    """
    修改vissl得到的vision transformer 名称
    
    当前在vissl上训练的模型没有blocks.0.attn.qkv.bias,计划先不对其进行改变

    
    v2: 2023/05/23修复mmsegmentation无法加载的问题,对于mmsegmentation而言,pretrained写在backbone下面,不需要backbone.的前缀
    
    """
    trunk=GetModelTrunk(visslViTcheckpointFilePath)
    trunk_new=deepcopy(trunk) # 需要进行深拷贝
    
    #---------------------------------------------------------------------------------------------
    """
    MMSegModel Layer Name:                                            VISSLModel Layer Name:
    backbone.cls_token                                                |class_token
    backbone.pos_embed                                                |pos_embedding
    backbone.patch_embed.projection.weight                              |patch_embed.proj.weight
    backbone.patch_embed.projection.bias                                |patch_embed.proj.bias
           
         backbone.layers.0.ln1.weight                                 |blocks.0.norm1.weight
         backbone.layers.0.ln1.bias                                   |blocks.0.norm1.bias
         backbone.layers.0.attn.attn.in_proj_weight                   |blocks.0.attn.qkv.weight
         backbone.layers.0.attn.attn.in_proj_bias # 一些模型为False     |# blocks.0.attn.qkv.bias # 为了方便,在VISSL训练时,此处设定为True
         backbone.layers.0.attn.attn.out_proj.weight                  |blocks.0.attn.proj.weight
    x12  backbone.layers.0.attn.attn.out_proj.bias                    |blocks.0.attn.proj.bias
         backbone.layers.0.ln2.weight                                 |blocks.0.norm2.weight 
         backbone.layers.0.ln2.bias                                   |blocks.0.norm2.bias 
         backbone.layers.0.ffn.layers.0.0.weight                      |blocks.0.mlp.fc1.weight 
         backbone.layers.0.ffn.layers.0.0.bias                        |blocks.0.mlp.fc1.bias 
         backbone.layers.0.ffn.layers.1.weight                        |blocks.0.mlp.fc2.weight
         backbone.layers.0.ffn.layers.1.bias                          |blocks.0.mlp.fc2.bias

    backbone.ln1.weight                                               |norm.weight
    backbone.ln1.bias                                                 |norm.bias
    """
    #----------------------------------------------------------------------------------------
    nameList=['cls_token','pos_embed','pos_embed.projection.','pos_embed.projection.'] # mmsegmentation中vit的patch的特征提取和投影层
    pre_layer='layers.' # mmsegmentation自注意力模块的统一前缀
    nameList_LN=['ln1.','ln2.'] # 归一化层的名称前缀
    nameList_qkv=['attn.attn.in_proj_weight','attn.attn.in_proj_bias','attn.attn.out_proj.'] # 自注意力层的名称前缀
    nameList_MLP=['ffn.layers.0.0.','ffn.layers.1.'] # 自注意力模块统一前缀

    LayerNum=12 # 自注意力模块统一的层数 编号0-11
    NameCount=0 # NameCount从0开始计数
    # 对于vissl的vit的每一层
    for name,value in trunk.items():
        if NameCount<2:
            """
            0:class_token->cls_token
            1:pos_embedding->pos_embed
            """
            # new_name=pre_all+nameList[NameCount] 
            new_name=nameList[NameCount] 
        elif NameCount<4:
            """
            2:patch_embed.proj.weight->pos_embed.projection.weight
            3:patch_embed.proj.bias->pos_embed.projection.bias
            """
            new_name=nameList[NameCount]+name.split('.')[-1]
        # 自注意力层名称替换
        elif NameCount<(12*LayerNum+4):
            """
            0 4+i*12:blocks.0.norm1.weight->layers.0.ln1.weight
            1 5:blocks.0.norm1.bias->       layers.0.ln1.bias
            2 6:blocks.0.attn.qkv.weight->  layers.0.attn.attn.in_proj_weight
            3 7:blocks.0.attn.qkv.bias->    layers.0.attn.attn.in_proj_bias     
            4 8:blocks.0.attn.proj.weight-> layers.0.attn.attn.out_proj.weight
            5 9:blocks.0.attn.proj.bias->   layers.0.attn.attn.out_proj.bias  
            6 10:blocks.0.norm2.weight->    layers.0.ln2.weight
            7 11:blocks.0.norm2.bias->      layers.0.ln2.bias
            8 12:blocks.0.mlp.fc1.weight->  layers.0.ffn.layers.0.0.weight
            9 13:blocks.0.mlp.fc1.bias->    layers.0.ffn.layers.0.0.bias 
            10 14:blocks.0.mlp.fc2.weight->  layers.0.ffn.layers.1.weight
            11 15:blocks.0.mlp.fc2.bias->    blayers.0.ffn.layers.1.bias
            """
            # 第n(从0开始计数)个注意力模块：n=int((NameCount-4)/12)
            n=int((NameCount-4)/12)
            i=NameCount-4-n*12 # 当前注意力模块的第i层
            if i<2:
                a=name.split('.')[-1]
                new_name=pre_layer+str(n)+'.'+nameList_LN[0]+name.split('.')[-1]
            elif i<4:
                new_name=pre_layer+str(n)+'.'+nameList_qkv[i-2]
            elif i<6:
                new_name=pre_layer+str(n)+'.'+nameList_qkv[-1]+name.split('.')[-1]
            elif i<8:
                new_name=pre_layer+str(n)+'.'+nameList_LN[1]+name.split('.')[-1]
            elif i<10:
                new_name=pre_layer+str(n)+'.'+nameList_MLP[0]+name.split('.')[-1]
            else:
                new_name=pre_layer+str(n)+'.'+nameList_MLP[1]+name.split('.')[-1]
        else:
            """
            norm.weight->backbone.ln1.weight
            norm.bias->  backbone.ln1.bias
            """
            new_name=nameList_LN[0]+name.split('.')[-1]
        trunk_new[new_name]=trunk_new.pop(name)
        NameCount+=1
    return trunk_new

def ChangeMMViTLayerName(mmViTcheckpointFilePath):
    """
    修改mmselfsup得到的vision transformer 名称

    2023/05/23修复mmsegmentation无法加载的问题,对于mmsegmentation而言,pretrained写在backbone下面,不需要backbone.的前缀
    """
    trunk=GetMMSelfModel(mmViTcheckpointFilePath)
    trunk_new=deepcopy(trunk) # 需要进行深拷贝
    #---------------------------------------------------------------------------------------------
    """
    不要backbone
    MMSegModel Layer Name:                                            MMSelfSupModel Layer Name:
    backbone.cls_token                                                |backbone.cls_token
    backbone.pos_embed                                                |backbone.pos_embed
    backbone.pos_embed.projection.weight                              |backbone.pos_embed.projection.weight
    backbone.pos_embed.projection.bias                                |backbone.pos_embed.projection.bias
           
         backbone.layers.0.ln1.weight                                 |backbone.layers.0.ln1.weight 
         backbone.layers.0.ln1.bias                                   |backbone.layers.0.ln1.bias
         backbone.layers.0.attn.attn.in_proj_weight                   |backbone.layers.0.attn.qkv.weight #
         backbone.layers.0.attn.attn.in_proj_bias # 一些模型为False     |backbone.layers.0.attn.qkv.bias #
         backbone.layers.0.attn.attn.out_proj.weight                  |backbone.layers.0.attn.proj.weight #
    x12  backbone.layers.0.attn.attn.out_proj.bias                    |backbone.layers.0.attn.proj.bias #
         backbone.layers.0.ln2.weight                                 |backbone.layers.0.ln2.weight 
         backbone.layers.0.ln2.bias                                   |backbone.layers.0.ln2.bias
         backbone.layers.0.ffn.layers.0.0.weight                      |backbone.layers.0.ffn.layers.0.0.weight 
         backbone.layers.0.ffn.layers.0.0.bias                        |backbone.layers.0.ffn.layers.0.0.bias 
         backbone.layers.0.ffn.layers.1.weight                        |backbone.layers.0.ffn.layers.1.weight
         backbone.layers.0.ffn.layers.1.bias                          |backbone.layers.0.ffn.layers.1.bias

    backbone.ln1.weight                                               |backbone.ln1.weight
    backbone.ln1.bias                                                 |backbone.ln1.bias
    """
    #----------------------------------------------------------------------------------------
    nameList=['cls_token','pos_embed','pos_embed.projection.','pos_embed.projection.'] # mmsegmentation中vit的patch的特征提取和投影层
    pre_layer='layers.' # mmsegmentation自注意力模块的统一前缀
    nameList_LN=['ln1.','ln2.'] # 归一化层的名称前缀
    nameList_qkv=['attn.attn.in_proj_weight','attn.attn.in_proj_bias','attn.attn.out_proj.'] # 自注意力层的名称前缀
    nameList_MLP=['ffn.layers.0.0.','ffn.layers.1.'] # 自注意力模块统一前缀

    LayerNum=12 # 自注意力模块统一的层数 编号0-11
    NameCount=0 # NameCount从0开始计数
    # 对于vissl的vit的每一层
    for name,value in trunk.items():
        if name.split('.')[0]=='backbone':
            if NameCount<2:
                """
                0:class_token->cls_token
                1:pos_embedding->pos_embed
                """
                # new_name=pre_all+nameList[NameCount] 
                new_name=name[9:]
            elif NameCount<4:
                """
                2:patch_embed.proj.weight->pos_embed.projection.weight
                3:patch_embed.proj.bias->pos_embed.projection.bias
                """
                new_name=name[9:]
            # 自注意力层名称替换
            elif NameCount<(12*LayerNum+4):
                """
                0 4+i*12:blocks.0.norm1.weight->layers.0.ln1.weight
                1 5:blocks.0.norm1.bias->       layers.0.ln1.bias
                2 6:layers.0.attn.qkv.weight->  layers.0.attn.attn.in_proj_weight #
                3 7:layers.0.attn.qkv.bias->    layers.0.attn.attn.in_proj_bias    # #
                4 8:layers.0.attn.proj.weight-> layers.0.attn.attn.out_proj.weight #
                5 9:layers.0.attn.proj.bias->   layers.0.attn.attn.out_proj.bias  #
                6 10:blocks.0.norm2.weight->    layers.0.ln2.weight
                7 11:blocks.0.norm2.bias->      layers.0.ln2.bias
                8 12:blocks.0.mlp.fc1.weight->  layers.0.ffn.layers.0.0.weight
                9 13:blocks.0.mlp.fc1.bias->    layers.0.ffn.layers.0.0.bias 
                10 14:blocks.0.mlp.fc2.weight->  layers.0.ffn.layers.1.weight
                11 15:blocks.0.mlp.fc2.bias->    blayers.0.ffn.layers.1.bias
                """
                # 第n(从0开始计数)个注意力模块：n=int((NameCount-4)/12)
                n=int((NameCount-4)/12)
                i=NameCount-4-n*12 # 当前注意力模块的第i层
                if i<2:
                    new_name=name[9:]
                elif i<4:
                    new_name=pre_layer+str(n)+'.'+nameList_qkv[i-2]
                elif i<6:
                    new_name=pre_layer+str(n)+'.'+nameList_qkv[-1]+name.split('.')[-1]
                elif i<8:
                    new_name=name[9:]
                elif i<10:
                    new_name=name[9:]
                else:
                    new_name=name[9:]
            else:
                """
                norm.weight->backbone.ln1.weight
                norm.bias->  backbone.ln1.bias
                """
                new_name=name[9:]
            NameCount+=1
        else:
            new_name=name
        trunk_new[new_name]=trunk_new.pop(name)
    return trunk_new


# def CheckModelName(model,checktype='resnet50'):
#     """
#     对操作完成的模型进行检查
#     """
#     if checktype=='resnet50':

#         print()
#     # TODO: other check type

def ChangeHeadName(visslCheckpointFilePath):
    """
    为了支持对simclr mlp层的加载参数的相关操作
    """
    heads=GetModelHeads(visslCheckpointFilePath)
    heads_new=deepcopy(heads) # 需要进行深拷贝
    for name,value in heads.items():
        #new_name=name.split('blocks.')[1]
        new_name='clp'+name
        heads_new[new_name]=heads_new.pop(name)
    return heads_new

def ChangeContrastiveCroplayerName(CheckpointFilePath,flag='moco_state'):
    """
    修改pcl得到的checkpoint名称
    """
    model=GetContrastiveCropModel(CheckpointFilePath,flag)

    trunk_tmp=deepcopy(model) # 需要进行深拷贝
    # 筛选得到需要的层
    keys=[]
    for name,value in model.items():
        if name.split('encoder_q.')[0]=='module.':
            keys.append(name)
    trunk_tmp = {k:trunk_tmp[k] for k in keys}

    trunk_new=deepcopy(trunk_tmp) # 需要进行深拷贝
    # 修改层名
    for tmp_name,value in trunk_tmp.items():
        new_name=tmp_name.split('encoder_q.')[1]
        trunk_new[new_name]=trunk_new.pop(tmp_name)
    return trunk_new