#马上插入main.py 中导入的包
from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from resnet_v2 import Model

#结束
import torch
from main import train
from main import test
from prune_helper import  compressed_model
from utils.Slim_file import updateBN
import config

"""
big_model_path = config.big_model_path
model = torch.load(big_model_path)['model']

train(model)

torch.save("new_path")
#基本框架就这样···
"""

if __name__ == '__main__':
    #首先应该加上bn的特殊梯度，然后重训练，为了简便，就在大模型的基础上重训练吧
    big_model_path = config.big_model_path

    #首先恢复train函数的训练参数
    train_parameters_str = 'model' + ',' + str(config.train_parameter[1]//5)
    for i in range(2,len(config.train_parameter)):
        train_parameters_str += ','
        train_parameters_str += str(config.train_parameter[i])

    train_parameters_str += ','
    train_parameters_str += 'updateBN'

    #恢复参数完毕

    #加载用户最初的模型
    model = torch.load(big_model_path)['model']

    #加上update之后的重训练
    print("加上update之后的重训练：")
    exec("train({})".format(train_parameters_str))

    #创建剪枝模型

    new_model = compressed_model(Model,model,config.cfg,config.ratio_prune,config.ratio_weightbn,config.version)
    print("已创建新的模型")
    #重训练
    #重新组织参数
    train_parameters_str = 'new_model' + ',' + str(config.train_parameter[1])
    for i in range(2,len(config.train_parameter)):
        train_parameters_str += ','
        train_parameters_str += str(config.train_parameter[i])
    #不用加update了
    #重训练

    print("fine-tune 压缩模型：")
    exec("train({})".format(train_parameters_str))

    #测试新模型的准确率
    print("压缩之后的模型准确率：")
    test_parameters_str = 'new_model'
    for i in range(1,len(config.test_parameter)):
        test_parameters_str +=','
        test_parameters_str += str(config.test_parameter[i])

    exec("test({})".format(test_parameters_str))


    #存储新的模型
    torch.save(new_model,"./logs/compressed_model.pth.tar")

    print("已存储压缩模型")








