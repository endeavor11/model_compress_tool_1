#这个文件应该用作转移除了主要block部分外其他的参数

#预计方法：
#
# (1)首先把CFG中最后一个模块last_CFG_module（应该是conv，对于preactivate来说）保留下来。
# (2)然后重新named_modules遍历模型中的模块。从last_CFG_module 往后，遇到一个有参数的模块就复制一下参数
# (3)当然这里需要辨识现在是哪个模块，目前常用的，可能在CFG之后的：bn，linear吧
# (4)先遍历全精度模型，遇到上述的模型就把参数保留下来，所有模型的参数都保留到一个列表里，如果该module有大于1种的参数，
#    就保留到tuple里面，然后遍历新模型，遇到一个上述的几种module的话，就把列表中相对应位置的参数复制过去就行了。
#
import torch
import torch.nn as nn
import copy

def transfer_other_weight_func(newmodel,slim):
    """

    :param newmodel:
    :param oldmodel:
    :return:
    """
    oldmodel_last_handled_module = slim.last_handled_module
    oldmodel_unhandled_modules = []

    flag = False
    for x,y in slim.model.named_modules():
        if(y==oldmodel_last_handled_module):
            flag = True
            continue

        if(flag==True):
            if(type(y) is nn.Linear):
                oldmodel_unhandled_modules.append(y)

            if(type(y) is nn.BatchNorm2d):
                oldmodel_unhandled_modules.append(y)


    #好像还得获取newmodel的最后一个处理的模块···
    newmodel_last_handled_module = None
    for x,y in newmodel.named_modules():
        if(type(y) is nn.Conv2d):
            newmodel_last_handled_module = y


    current_num = 0
    flag = False
    for x,y in newmodel.named_modules():
        if(y==newmodel_last_handled_module):
            flag = True
            continue

        if(flag==True):
            if(type(y) is nn.Linear):
                y.weight.data = oldmodel_unhandled_modules[current_num].weight.data.clone()
                if(y.bias is not None):
                    y.bias.data = oldmodel_unhandled_modules[current_num].bias.data.clone()
                current_num += 1

            if(type(y) is nn.BatchNorm2d):
                y.weight.data = oldmodel_unhandled_modules[current_num].weight.data.clone()
                y.bias.data = oldmodel_unhandled_modules[current_num].weight.data.clone()
                y.running_mean.data = oldmodel_unhandled_modules[current_num].running_mean.data.clone()
                y.running_var.data = oldmodel_unhandled_modules[current_num].running_var.data.clone()
                current_num += 1


