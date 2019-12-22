import torch
import torch.nn as nn
import utils



"""
创建一个函数，预计只用调用这一个函数，就可以返回很好的模型了
"""


def compressed_model(the_class,model,CFG,ratio_prune,ratio_weightbn,version):
    slim = utils.Slim_file.Slim(model,CFG,ratio_prune,ratio_weightbn)
    new_config = slim.prune(version)
    new_model = the_class(new_config)
    #utils.transfer_weight.transfer_weight(new_model,slim)
    #utils.transfer_other_weight.transfer_other_weight(new_model,slim)
    utils.transfer_weight.transfer_weight_func(new_model,slim)
    utils.transfer_other_weight.transfer_other_weight_func(new_model,slim)

    return new_model

if __name__ == '__main__':
    print(utils.transfer_weight.transfer_weight_func)