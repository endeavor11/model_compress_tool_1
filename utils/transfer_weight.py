#定义参数转移函数
import torch
import torch.nn as nn
import numpy as np
import copy


def transfer_weight_func(new_model,slim):
    """

    :param new_model:
    :param slim:
    :return:
    """

    new_model_conv_list = []
    new_model_bn_list = []

    last_bn = None
    for x,y in new_model.named_modules():
        if(type(y) is nn.BatchNorm2d):
            last_bn = y
        elif(type(y) is nn.Conv2d):
            new_model_conv_list.append(y)
            new_model_bn_list.append(last_bn)
            last_bn = None

    #获取新模型的conv-list和bn-list已结束
    #复制bn的参数
    for i in range(len(slim.bn_list)):
        new_model_bn_list[i].weight.data = slim.bn_list[i].weight.data[slim.index_of_retain[i]].clone()
        new_model_bn_list[i].bias.data = slim.bn_list[i].bias.data[slim.index_of_retain[i]].clone()
        new_model_bn_list[i].running_mean.data = slim.bn_list[i].running_mean.data[slim.index_of_retain[i]].clone()
        new_model_bn_list[i].running_var.data = slim.bn_list[i].running_var.data[slim.index_of_retain[i]].clone()

    #复制conv的参数
    for i in range(len(slim.conv_list)):
        pre_bn = i
        post_bn = slim.conv_correlation_bns[i][1]  #有可能是None 每个conv对应的是[pre,post]  [4,5]这样的

        pre_bn_index = slim.index_of_retain[pre_bn]
        if(post_bn==None):  #如果是None，就说明是最后一个conv了，一个也不减
            post_bn_index = []
            for j in range(slim.conv_list[i].out_channels):
                post_bn_index.append(j)
        else:
            post_bn_index = slim.index_of_retain[post_bn]

        #print('post:',post_bn_index)
        #print('pre:',pre_bn_index)
        #print('size:',slim.conv_list[i].weight.data.size())
        #print('====================================================')
        #new_model_conv_list[i].weight.data = slim.conv_list[i].weight.data[post_bn_index,pre_bn_index,:,:].clone()
        temp = slim.conv_list[i].weight.data.clone().numpy()
        temp = temp[np.ix_(post_bn_index,pre_bn_index)]  #必须转换成numpy才行，因为广播说是不支持用list索引两个轴，但是用了
                                                         #numpy 的 ix_ 函数就可以了
        temp = torch.from_numpy(temp)
        new_model_conv_list[i].weight.data = temp



        if(new_model_conv_list[i].bias is not None):
            new_model_conv_list[i].bias.data = slim.conv_list[i].bias.data[post_bn_index].clone()



    #完毕了吧，conv和bn的参数都搞定了

