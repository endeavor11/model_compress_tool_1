import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import numpy as np
import copy

"""
真正的剪枝的代码

        cfg示例：
            cfg = [16,  {0:[16,16,64],1:[64]},   [[16,16,64],[64]],  [16,16,64],
                        {0:[32,32,128],1:[128]}, [32,32,128], 'Max' [[32,32,128],[128]],
                        {0:[64,64,256],1:[256]}, [64,64,256],  [64,64,256],'Avg',16,16,32,{0:[16,16,64]},'FC']

            也有可能有 {0:[16,16,64]}   就是说不论是dict还是list，都有可能是双路或者单路，不一定，需自己判断

1、有没有Max，Avg应该都有能力去解决，甚至有的max写出来，有的没有写出来，也应该解决.Max，Avg不能放在花括号和方括号里面

2、要剪枝的只有conv2d

3、分两类，preactivate和posactivate
先写preactivate，因为这是更新的方法

4、如果是字典，并且里面只有一个元素，就说明没有旁支，那么其实去掉这花括号也是无所谓的。

5、最新：第一层conv前即使不连接bn也行，这样的话，就是一个None

6、如果模型的前几层是conv，然后几层conv后连接了其他非常特殊的操作，可以选择只压缩conv的部分，就是把几层conv写进cfg就行了，
前面没有特殊的操作，完全还可以压缩，就算非常特殊的操作之后还有其他的conv也没关系了，因为我只按照cfg里面的东西进行压缩，就算
convlist bnlist多了很多东西也无所谓，根本就不用

7、有连续依赖问题，就是连着两个list！！这样就会有好几个conv必须有相同数目的通道了，得，强。
但是自己的那个字典，只返回紧挨着的，自己使用的时候应该要全部判断才行

8、目前所有的self.xxx 都是在init里面定义的，函数里面没有私自定义self.成员

9、成员变量说明
self.parallel_bn_dict = {1:4,11:14,21:24}  输入主干路的序号，输出旁路的序号

"""


class Slim():
    def __init__(self,model,CFG,ratio,ratio_parabn):
        """

        :param CFG:
        """
        self.model = model
        self.CFG = copy.deepcopy(CFG)
        self.CFG_canceled = self.cancel_max_avg(CFG)
        self.CFG_serial_number = self.convert_conv_to_serial_number(self.CFG_canceled)
        self.CFG_modules, self.bn_list, self.conv_list = self.convert_serial_number_to_bn_conv(model,
                                                                                               self.CFG_serial_number)
        #self.CFG_modules 没有用上，完全可以删去
        self.output_channel_num = self.return_output_channel_convs()
        self.parallel_bn_dict, self.parallel_bn_list = self.return_parallel_bn(self.CFG_serial_number) # 这个字典是
                                                                                                       # int->int 的
        self.constrained_bn = self.return_constrained_bn(self.CFG_serial_number)
        self.bn_flags = self.return_bn_flags()
        self.ratio = ratio
        self.ratio_parabn = ratio_parabn
        # threshold = self.get_threshold_v1() #获取threshold的函数还是在prune函数里面再运行吧因为感觉会变不少次
        # threshold = self.get_threshold_v2(self.ratio_parabn) #第二个版本，考虑并联bn不同分支权重的情况
        # threshold不是self变量了，在prune函数里面自己去调用这两个函数去使用吧
        self.bn_correlation_convs = self.return_bn_correlation_convs(self.CFG_serial_number)

        self.last_list_head = self.return_last_list_head()
        self.first_list_head = self.return_first_list_head()

        self.conv_correlation_bns = self.return_conv_correlation_bns()

        self.last_handled_module = self.conv_list[-1]  #这里保留一下最后一个处理的模块，为了方便之后处理这个模块之后又
                                                       #有参数的层


    def cancel_max_avg(self,CFG):
        # 删去cfg中的Maxpool和Avgpool，因为这些对通道数没有啥作用，并且现在规定，这两个不能待在[]和{}里面
        CFGs = copy.deepcopy(CFG)
        new_CFG = []
        for i in CFGs:
            if(i=='Max' or i=='Avg'):
                pass
            else:
                new_CFG.append(i)
        return new_CFG




    def convert_conv_to_serial_number(self,CFG):
        # 把对应的conv通道数变成编号，编号规则是按照named-modules的返回顺序进行的，也就是若有旁支，先编号主干路，也就是字典或者列表
        # 中的第一个元素
        #new_CFG = CFG[:] # 赋值过来的新的列表，对这个东西进行操作
        new_CFG = copy.deepcopy(CFG)
        current_num = 0
        for i in range(len(new_CFG)):
            if(type(new_CFG[i]) is int):
                new_CFG[i] = current_num
                current_num += 1
            elif(type(new_CFG[i]) is list):
                if(type(new_CFG[i][0]) is int): # new_CFG[i] : [16,16,32] 这里不能再用长度来判断类型了，因为有可能有的
                                                # []里面就是2个元素
                    for j in range(len(new_CFG[i])):
                        new_CFG[i][j] = current_num
                        current_num += 1
                else: # 有旁支module的情况 new_CFG[i] = [[32,32,32],[32]] 可以肯定的是目前来说，肯定是两个分支，
                      # 先不考虑多个分支的情况
                    # 处理主干路
                    for j in range(len(new_CFG[i][0])):
                        new_CFG[i][0][j] = current_num
                        current_num += 1

                    for j in range(len(new_CFG[i][1])):
                        new_CFG[i][1][j] = current_num
                        current_num += 1

            elif(type(new_CFG[i]) is dict):
                if(len(new_CFG[i])==1): # new_CFG[i] = {0:[16,16,64]}
                    for j in range(len(new_CFG[i][0])): # new_CFG[i][0] = [16,16,64]
                        new_CFG[i][0][j] = current_num
                        current_num += 1
                else: # 有旁支module的情况 new_CFG[i] = {0:[16,16,64],1:[64]}
                    for j in range(len(new_CFG[i][0])):
                        new_CFG[i][0][j] = current_num
                        current_num += 1
                    for j in range(len(new_CFG[i][1])):
                        new_CFG[i][1][j] = current_num
                        current_num += 1

        return new_CFG







    def convert_serial_number_to_bn_conv(self,model,CFG):
        # 输入是conv编号的CFG，现在要把(BN,Conv2d)代替原先的数字 注意是编号之后的CFG！不是最开始用户输入的CFG了
        # 返回 bn列表，conv列表，以及把序号替换成bn，conv元组的CFG
        #new_CFG = CFG[:]
        new_CFG = copy.deepcopy(CFG)
        conv_list = []
        bn_list = []

        last_bn = None
        for x,y in model.named_modules():
            if(type(y) is nn.BatchNorm2d):
                last_bn = y
            if(type(y) is nn.Conv2d):
                conv_list.append(y)
                bn_list.append(last_bn)
                last_bn = None # 这样可以保证，每个bn只能使用一次，就是说，如果conv前面没有bn的话，填进去的东西就是None，有点意思
                               # 但是还是不能bn后面连接什么奇怪的操作比如 bn conv bn conv bn view shuffle conv
                               # 因为这么一套中间操作有可能
                               # channel就对不上号了！数量也可能不匹配了

        # 又要遍历一次CFG了
        for i in range(len(new_CFG)):                # 有三种类型 int list dict list和dict第一个元素还有可能是[] 类型，
                                                     # 最好不要用list的元素
            if(type(new_CFG[i]) is int):             # 个数来判断有没有旁支的modules了，因为有可能就2个卷积层，然后以判断，
                                                     # 还以为长度为2的是一个有旁支modules# 的东西呢
                new_CFG[i] = (bn_list[CFG[i]],conv_list[CFG[i]])
            elif(type(new_CFG[i]) is list):
                if(type(new_CFG[i][0]) is int): # CFG[i] = [16,16,32]
                    for j in range(len(new_CFG[i])):
                        new_CFG[i][j] = (bn_list[CFG[i][j]],conv_list[CFG[i][j]])
                elif(type(new_CFG[i][0] is list)): # CFG[i] = [[16,16,32],[32]]
                    for j in range(len(new_CFG[i][0])): # new_CFG[i][0] = [16,16,32]
                        new_CFG[i][0][j] = (bn_list[CFG[i][0][j]],conv_list[CFG[i][0][j]])
                    for j in range(len(new_CFG[i][1])): # new_CFG[i][1] = [16,16,32]
                        new_CFG[i][1][j] = (bn_list[CFG[i][1][j]],conv_list[CFG[i][1][j]])
            elif(type(new_CFG[i]) is dict): # 但是字典的判断得用len了，因为字典的两个元素都是list
                if(len(new_CFG[i])==1): # 没有旁支的 {0:[16,16,32]}
                    for j in range(len(new_CFG[i][0])): # new_CFG[i][0] = [16,16,32]
                        new_CFG[i][0][j] = (bn_list[CFG[i][0][j]],conv_list[CFG[i][0][j]])
                elif(len(new_CFG[i])==2): # {0:[16,16,64],1:[64]}
                    for j in range(len(new_CFG[i][0])):
                        new_CFG[i][0][j] = (bn_list[CFG[i][0][j]],conv_list[CFG[i][0][j]])
                    for j in range(len(new_CFG[i][1])):
                        new_CFG[i][1][j] = (bn_list[CFG[i][1][j]],conv_list[CFG[i][1][j]])


        return new_CFG,bn_list,conv_list






    def return_parallel_bn(self,CFG):   # 参数是序号CFG
                                   # 对cfg是字典的情况也收录，其实主要就是这个情况，而且之后还想探索两个并列bn加权系数不同的话
                                   # 对剪枝效果的影响呢
                                   # 所以字典也返回！不然这个函数真没啥用了
                                   # 用字典来存储 ，将来寻找有没有并列的时候就很方便
                                   # 采用遍历cfg的方法
                                   # 现在是对称的了！1->2 的同时，肯定存在2->1
        parallel_bn_dict = {}
        parallel_bn_list = {}
        for i in CFG:
            if(type(i) is int):
                continue
            elif(type(i) is list):
                if(type(i[0]) is list): # i = [[1,2,3],[4]]
                    parallel_bn_list[i[0][0]] = i[1][0]
            elif(type(i) is dict):
                if(len(i)==2): # i = {0:[1,2,3],1:[4]}
                    parallel_bn_dict[i[0][0]] = i[1][0]


        # 需要做一个对称处理，旁路也可以映射到主干路，虽然好像没啥必要，但为了完备点还是做吧，因为剪枝的时候，应该是按照bn-list
        # 里面的顺序，肯定是先处理的主干路，然后处理主干路的时候，监测到右旁路对应的bn，这样子就顺便处理了旁路的bn。
        new_parallel_bn_dict = {}
        new_parallel_bn_list = {}
        for x,y in parallel_bn_list.items():
            new_parallel_bn_list[x] = y
            new_parallel_bn_list[y] = x

        for x,y in parallel_bn_dict.items():
            new_parallel_bn_dict[x] = y
            new_parallel_bn_dict[y] = x

        return new_parallel_bn_dict,new_parallel_bn_list





    def return_constrained_bn(self,CFG):  #最重要的：dict的那些根本就不会考虑！
        """
        最关键的一个函数。只针对list类型的第一个bn有效果，因为就是list才限制维度的，如果是字典就放手了···
        相互制约的第一个bn是每个list的开头的第一个(可能有并列)的bn，然后返回这个list结束之后的第一个bn(也可能有并列的情况)

        确实是只求了list的相关的部分，dict既然用户没要求求，我就没求···之后可以补充···

        剪枝的时候还是要注意，最开始的输入维度不减以及输出维度不减
        :param CFG: 应该是self.CFG_serial_number
        :return: 字典 -键是bn或者bn元组，取决于有没有并列的bn，值还是bn或者bn元组
        """
        constrained_bn = {}
        for i in range(len(CFG)): # 不能用遍历的方法啦，因为要往后取好几个module呢，只能用index的形式

            if(i==len(CFG)-1): # 说明是最后一个了输出维度不能变化，所以这个就不剪枝了
                               # 这里的措施其实没啥用，因为你最后一个block不进行下面的东西了，是因为下面就在没有list了
                               # 但是本身最后一个block的第一个通道的bn还是有可能被上面的list的通道关联到。但是这步也不能少
                break
            bn1 = None
            bn2 = None

            if(type(CFG[i]) is list): # CFG[i]
                if(type(CFG[i][0]) is int): #CFG[i] = [3,4,5]
                    bn1 = CFG[i][0]
                elif(type(CFG[i][0]) is list): # CFG[i] = [[3,4,5],[6]]
                    bn1 = CFG[i][0][0]
                    bn2 = self.parallel_bn_list[bn1] # 其实直接从CFG[i]里面也能获取，哈哈哈
                # 该考察i+1了，看是int 还是 list 还是 dict 同时也要注意到i+1也有旁支module的情况

                bn1_after = None
                bn2_after = None

                if(type(CFG[i+1]) is int): # CFG[i+1] = 5 这种
                    pass
                elif(type(CFG[i+1]) is list):
                    if(type(CFG[i+1][0]) is int): # CFG[i+1] = [3,4,5]
                        bn1_after = CFG[i+1][0]
                    elif(type(CFG[i+1][0]) is list):# CFG[i+1] = [[3,4,5],[6]]
                        bn1_after = CFG[i+1][0][0]
                        bn2_after = CFG[i+1][1][0]
                elif(type(CFG[i+1]) is dict):
                    if(len(CFG[i+1])==1): # CFG[i+1] = {0:[3,4,5]}
                        bn1_after = CFG[i+1][0][0]
                    elif(len(CFG[i+1])==2):# CFG[i+1] = {0:[3,4,5],1:[6]}
                        bn1_after = CFG[i+1][0][0]
                        bn2_after = CFG[i+1][1][0]
                # bn1_after bn2_after 构造完毕

                if(bn2==None): # 都是序号
                    if(bn2_after==None):
                        constrained_bn[bn1] = bn1_after
                    else:
                        constrained_bn[bn1] = (bn1_after,bn2_after)
                else:
                    if (bn2_after == None):
                        constrained_bn[(bn1,bn2)] = bn1_after
                    else:
                        constrained_bn[(bn1,bn2)] = (bn1_after, bn2_after)


        return constrained_bn

    def return_bn_flags(self):
        """
        返回尺寸和conv-list，bn-list相同大小的列表，最开始每个元素都是false，说明还没有处理过，如果变成True了。就说明处理过
        :return:
        """
        bn_flags = []
        for i in range(len(self.bn_list)):
            bn_flags.append(False)

        return bn_flags


    def return_bn_correlation_convs(self,CFG):
        """
        还是返回索引！！不是真的conv！没必要！这样也方便调试
        ok了，在test-辅助函数2里面有测试，感觉还不错呢
        :param CFG:索引CFG -- serial-number
        :return: 应该返回list，每个元素是一个元组，元素可能是 1 (conv1,conv2)  2 ((conv1,conv4),conv5) 因为前驱有可能是两个conv的

        """
        def get_last(index,CFG):
            """
            求list或者dict的第一个元素的前驱conv的函数，要设计到当前list或者dict的上一个int 或者 list 或者 dict的最后一个conv的序号
            有两个的话就返回元组

            返回的还是索引！！

            :param index:
            :param CFG:
            :return:
            """

            if(i==0):  #如果是第一个conv的话，就返回None
                return None

            if(type(CFG[index-1]) is int):
                return CFG[index-1]
            elif(type(CFG[index-1]) is list):
                if(type(CFG[index-1][0]) is int):  # CFG[index-1] = [2,3,4]
                    return CFG[index-1][-1]
                elif(type(CFG[index-1][0]) is list):  # CFG[index-1] = [[2,3,4],[5]]
                    return (CFG[index-1][0][-1],CFG[index-1][1][-1])
            elif(type(CFG[index-1]) is dict):
                if(len(CFG[index-1])==1):  #CFG[index-1] = {0:[4,5,6]}
                    return CFG[index-1][0][-1]
                else:                      #CFG[index-1] = {0:[4,5,6],1:[7]}
                    return (CFG[index-1][0][-1],CFG[index-1][1][-1])


        # 内嵌函数定义完毕
        bn_correlation_convs = []
        #底下的这个遍历的顺序和当初把conv通道数变成序号的过程是一样的，现在再这么进行一次，所以压入bn-correlation-conv的顺序和编号的顺序
        #是一样的，所以可以达到对应的效果···这么说当初是不是应该在转化成序号的地方就把这个功能做了呢···
        for i in range(len(CFG)):
            if(type(CFG[i]) is int):  #就是一个int，当然要用get-last了
                temp = get_last(i,CFG)
                temp = (temp,CFG[i])  #后继conv就是自身的这序号！
                bn_correlation_convs.append(temp)
            elif(type(CFG[i]) is list):
                if(type(CFG[i][0]) is int):  #CFG[i] = [4,5,6]
                    # 还得for一下，遍历所有的元素，都用len来索引，不然不方便
                    for j in range(len(CFG[i])):
                        if(j==0):
                            temp = get_last(i,CFG)
                            temp = (temp,CFG[i][j]) # j==0
                            bn_correlation_convs.append(temp)
                        else:
                            temp = CFG[i][j-1]
                            temp = (temp,CFG[i][j])
                            bn_correlation_convs.append(temp)
                elif(type(CFG[i][0]) is list):  #CFG[i] = [[4,5,6],[7]]
                    #先处理主干路
                    for j in range(len(CFG[i][0])):  #CFG[i][0] = [4,5,6]
                        if(j==0):
                            temp = get_last(i,CFG)
                            temp = (temp,CFG[i][0][j])
                            bn_correlation_convs.append(temp)
                        else:
                            temp = CFG[i][0][j-1]
                            temp = (temp,CFG[i][0][j])
                            bn_correlation_convs.append(temp)
                    for j in range(len(CFG[i][1])):  #CFG[i][1] = [7]
                        if(j==0):
                            temp = get_last(i,CFG)
                            temp = (temp,CFG[i][1][j])
                            bn_correlation_convs.append(temp)
                        else:
                            temp = CFG[i][1][j-1]
                            temp = (temp,CFG[i][1][j])
                            bn_correlation_convs.append(temp)


            elif(type(CFG[i]) is dict):
                if(len(CFG[i])==1):  #CFG[i] = {0:[3,4,5]}
                    for j in range(len(CFG[i][0])):
                        if(j==0):
                            temp = get_last(i,CFG)
                            temp = (temp,CFG[i][0][j])
                            bn_correlation_convs.append(temp)
                        else:
                            temp = CFG[i][0][j-1]
                            temp = (temp,CFG[i][0][j])
                            bn_correlation_convs.append(temp)
                elif(len(CFG[i])==2):  #CFG[i] = {0:[3,4,5],1:[6]}
                    for j in range(len(CFG[i][0])):
                        if(j==0):
                            temp = get_last(i,CFG)
                            temp = (temp,CFG[i][0][j])
                            bn_correlation_convs.append(temp)
                        else:
                            temp = CFG[i][0][j-1]
                            temp = (temp,CFG[i][0][j])
                            bn_correlation_convs.append(temp)

                    for j in range(len(CFG[i][1])):
                        if(j==0):
                            temp = get_last(i,CFG)
                            temp = (temp,CFG[i][1][j])
                            bn_correlation_convs.append(temp)
                        else:
                            temp = CFG[i][1][j-1]
                            temp = (temp,CFG[i][1][j])
                            bn_correlation_convs.append(temp)

        return bn_correlation_convs

    def get_threshold_v1(self):
        """
        把所有的bn的参数拍格列，然后求出权重来
        self.ratio 越小，剪枝的就越少

        每个并行通道的重要性现在认为一样，最简单的版本

        :return:
        """

        threshold = 0


        # 底下这段代码是不对的，因为每个bn独立地剪枝了，然而应该考虑到通道之间的约束的 然而不能说不对吧···到时候都应该去试试效果的
        # 这个应该作为基本的一个对照组
        all_value_bn = []
        all_num = 0
        for i in self.bn_list: # 注意这个bn-list中的每一个bn都严格是conv前面的那个bn，独胆的bn，avg前面的bn都没有被包括在内
            all_value_bn += list(torch.abs(i.weight.data))
            all_num += list(i.weight.data.size())[0]

        thresh_index = all_num * self.ratio
        all_value_bn.sort() # 列表中每个元素都是tensor(4.0)类似这样的，并且直接sort就可以进行排序，这个已经验证过。升序
        threshold = all_value_bn[int(thresh_index)]

        return threshold

    def get_threshold_v2(self,ratio):
        """

        :param ratio: 主干路和旁路权重的分配情况 应当是主干路为主吧 ratio为0.7说明主干路bn的权值为0.7
        :return:
        """
        threshold = 0.0
        all_value_bn = []
        all_num = 0
        bn_flag_thresh = [False for x in self.bn_list] # (⊙o⊙)… 如果是false表明这个bn的参数还没有被考虑进去
        for i in range(len(self.bn_list)):
            if(bn_flag_thresh[i]==True):
                continue

            bn_flag_thresh[i] = True

            bn1 = self.bn_list[i]
            bn2_parallel = None

            if(i in self.parallel_bn_list.keys()):
                bn2_parallel = self.bn_list[self.parallel_bn_list[i]]
                bn_flag_thresh[self.parallel_bn_list[i]] = True
            elif( i in self.parallel_bn_dict.keys()):
                bn2_parallel = self.bn_list[self.parallel_bn_dict[i]]
                bn_flag_thresh[self.parallel_bn_dict[i]] = True

            # 该计算值了
            if(bn2_parallel==None):
                all_value_bn += list(torch.abs(bn1.weight.data))
                all_num += list(bn1.weight.data.size())[0]
            else:
                value1 = torch.abs(bn1.weight.data)
                value2 = torch.abs(bn2_parallel.weight.data)
                value_mean = value1*ratio + value2*(1-ratio)

                all_value_bn += list(value_mean)
                all_value_bn += list(value_mean)
                all_num += list(value_mean.size())[0]*2
        # 计算threshold

        all_value_bn.sort()
        thresh_index = all_num * self.ratio
        threshold = all_value_bn[int(thresh_index)]

        return threshold







    def pruner_phase1(self,version):
        """
        最主要的剪枝函数上线···
        甚至不需要model参数了，因为约束情况自己都已经知道了，bn，conv也拿出来list了

        遍历所有的bn（从bn-list里面遍历），先找有没有并列的bn，如果没有，就单独一个数，往约束字典里寻找有没有对应的键值对，有的话，
        再把这个新得到的约束conv的序号往约束字典里面
        寻找，有的话就加到列表(这个列表表示这次剪枝需要考虑的所有的conv)里面。一直这样知道没有对应的键值对。最后的结果是这样子的
        [(1,2),(4,5),6,(7,8),9,10,(11,12)]
        然后取一个元素，如果是int，直接看剪枝几个。如果是元组，比如(1,2),则用求threshhold的加权系数self.ratio_parabn来加权并列bn
        和thresh比，求出该层该剪枝几个通道，添加到列表
        再取下一个元素···

        结果：[(4,4),(5,5),3,(9,9),3,5,(1,1)] (这个表示每个conv需要剪枝的通道数)

        (1)然后求平均 结果：4+4+5+5+3+9+9+3+5+1+1  /  11 = 49/11 = 4 所以就剪枝4个通道
        (2)有可能上面求出来的通道数太大，有的层根本没这么多通道，那么就让最小的数字当做需要剪枝的数量 用上面的list为例，就是1.嗯···
        剪枝通道数-->  [(4,4),(4,4),4,(4,4),4,4,(4,4)]           ||          [(1,1),(1,1),1,(1,1),1,1,(1,1)]
        bn序号   -->  [(1,2),(4,5),6,(7,8),9,10,(11,12)] （一次剪枝约束了7层···）

        然后找最小的几个权值，记录-->
                     [[0,1,2,3],[0,5,9,11],[4,5,6,7],[66,67,68,68],[1,2,3,4],[9,9,9,9],[1,2,3,4]] 7个元素，有的有平行bn的话
                                                                                                  共用一个剪枝索引列表，因为
                                                                                                  要剪枝的通道必须是一样的通
                                                                                                  道

        然后上面这些bn的flag_prune = True
        有了上面这个应该可以创建新的config列表了啊

        接下来应该写得出新config的函数 原理：遍历bn-list 找所有的依赖通道-->决定减去几个通道-->减去哪几个通道-->该bn的上一个conv输出通道数减去该
        个bn减去通道数就是新的输出通道数
        (1)减去哪几个通道需要永久保留，list存储，注意需要提前放n(n是conv的数量)个东西，以便之后索引list[i] 因为剪枝不是按bn-list来的，会
           提前处理有约束的通道
        (2)输出通道数需要永久保留，原因同上，剪枝不是按照顺序来的
        (3)

        version: 阈值用v1的方法来使v2的方法，应该是一个字符串 "v1"  ||  "v2"
        :return:
        num_of_needed_pruned = [4,4,4,4,5,5,5,5]
        index_of_pruned = [[0，1，2],[2，3，4],[5，7，8],[1，4，6],[6，7，8],[8，9，0]···]
        index_of_retain = [[3,4,5],[0,1,5],[0,1,2,3,4],[0,2,3,5],[0,1,2,],[0,1,2,3]···]
        """
        def get_pruned_num_v1(threshold,bns):
            """
            加权系数是0.5。
            计算bns着一个bn或者两个bn，按照门限是threshold来剪枝的话，需要剪枝的数量
            :param threshold:
            :param bns:
            :return: int 或者tuple 类似于 (4,4)
            """
            if (type(bns) is int):
                vars = torch.abs(self.bn_list[bns].weight.data)
                return list(vars[vars < threshold].view(-1).size())[0]  # 求出大于某个数的好方法啊！
            elif (type(bns) is tuple):
                vars1 = torch.abs(self.bn_list[bns[0]].weight.data)
                vars2 = torch.abs(self.bn_list[bns[1]].weight.data)
                vars = 0.5 * vars1 + 0.5 * vars2
                return (list(vars[vars < threshold].view(-1).size())[0],list(vars[vars < threshold].view(-1).size())[0])
            else:
                print("second parameter is not a int or a tuple!!")


        def get_pruned_num_v2(threshold,bns):
            """
            计算bns着一个bn或者两个bn，按照门限是threshold来剪枝的话，需要剪枝的数量
            加权系数由初始化的时候的输入值来决定
            :param threshold:
            :param bns: 一个bn或者两个bn，一个bn的话，就直接是int。两个bn就是元组(int,int) 代表索引 eg: 5  ||   (4,5)
            :return:
            """
            if(type(bns) is int):
                vars = torch.abs(self.bn_list[bns].weight.data)
                return list(vars[vars<threshold].view(-1).size())[0]  #求出大于某个数的好方法啊！
            elif(type(bns) is tuple):
                vars1 = torch.abs(self.bn_list[bns[0]].weight.data)
                vars2 = torch.abs(self.bn_list[bns[1]].weight.data)
                vars = self.ratio_parabn*vars1 + (1-self.ratio_parabn)*vars2
                return (list(vars[vars < threshold].view(-1).size())[0],list(vars[vars < threshold].view(-1).size())[0])
            else:
                print("second parameter is not a int or a tuple!!")


        def get_pruned_num_of_all_constrained_bns_v1(num_of_needed_pruned):
            """
            获取了一个类似于[(4,4),(5,5),3,(9,9),3,5,(1,1)] 这样的东西，现在要找出这里面的最小的元素
            :param num_of_needed_pruned:
            :return: int 按照上面的来说，就是返回1
            """
            num_all = []
            for i in num_of_needed_pruned:
                if(type(i) is int):
                    num_all.append(i)
                elif(type(i) is tuple):
                    num_all.append(i[0])
                else:
                    print("输入的剪枝数量的参数既不是int也不是tuple！！")

            num_all.sort()
            return num_all[0]


        def get_pruned_and_retain_index_v1(num_real,bns):
            """
            按照门限是threshold，然后bns是一个或者两个bn （5 ||  (4,5) 看需要减去哪几个通道 类型是通道或者元组 ）
            :param num_real: int eg：2
            :param bns: int 或者 tuple 里面是int   eg：5 或者 (4,5)
            :return: bns是int ： eg: [0,3,6,7] 和 [1,2,4,5]
                     bns是tuple：eg: [0,3,6,7] 和 [1,2,4,5]
                     不管输入的是一个通道还是两个通道，返回的都是表示一个通道的量·
            """
            if(type(bns) is int):

                num_of_channel = self.bn_list[bns].weight.data.size()[0]  #通道数，eg:num_of_channel = 16

                value1 = torch.abs(self.bn_list[bns].weight.data).numpy()  #不然怕不是直接对bn里面的参数进行操作了··
                value2 = copy.deepcopy(torch.abs(self.bn_list[bns].weight.data).numpy())  #value2用来排序后找出门限值，value1抓取indexes
                value2.sort()


                temp_thresh = value2[num_real-1]  #比如说剪枝三个 这里获取value[2] 然后获取所有小于等于这个的下标，再取前3个，就OK了
                if (num_real == 0):
                    temp_thresh = -1.0
                indexes_pruned = np.where(value1<=temp_thresh)[0]  #因为value1是一维的话，这个返回的是(array([0,2,3], dtype=int64),)
                indexes_pruned = indexes_pruned[0:num_real]  #num_real所在的下标是不能取的
                #如果用np.where取≥temp_thresh的话，不一定对，考虑好多个bn的通道数的权重一样的话， 这样有可能剪枝的和留下的重合了··
                #所以应该从所有的index中拿出去上面的那些
                #方法：把上面的值转化成list，然后，用range遍历索引，找出所有的值不在那个list中的数字，就收纳进去···很傻的方法···
                indexes_pruned = list(indexes_pruned)  #indexes_pruned = [0,3,6,7] (such as)
                indexes_retain = []

                for i in range(num_of_channel):
                    if(i not in indexes_pruned):
                        indexes_retain.append(i)
                #indexes_retain = [1,2,4,5]
                return indexes_pruned,indexes_retain


            elif(type(bns) is tuple):
                num_of_channel = self.bn_list[bns[0]].weight.data.size()[0]
                value1 = 0.5*torch.abs(self.bn_list[bns[0]].weight.data).numpy()+0.5*torch.abs(self.bn_list[bns[1]].weight.data).numpy()
                value2 = copy.deepcopy(value1)
                value2.sort()

                temp_thresh = value2[num_real-1]
                if (num_real == 0):
                    temp_thresh = -1.0
                indexes_pruned = np.where(value1<=temp_thresh)[0]
                indexes_pruned = indexes_pruned[0:num_real]

                indexes_pruned = list(indexes_pruned)
                indexes_retain = []

                for i in range(num_of_channel):
                    if(i not in indexes_pruned):
                        indexes_retain.append(i)
                return indexes_pruned,indexes_retain

        def get_pruned_and_retain_index_v2(num_real,bns):
            """
            注释和v1的版本的一样
            :param num_real:
            :param bns:
            :return:
            """
            if (type(bns) is int):

                num_of_channel = self.bn_list[bns].weight.data.size()[0]  # 通道数，eg:num_of_channel = 16

                value1 = torch.abs(self.bn_list[bns].weight.data).numpy()  # 不然怕不是直接对bn里面的参数进行操作了··
                value2 = copy.deepcopy(
                    torch.abs(self.bn_list[bns].weight.data).numpy())  # value2用来排序后找出门限值，value1抓取indexes
                value2.sort()

                temp_thresh = value2[num_real - 1]  # 比如说剪枝三个 这里获取value[2] 然后获取所有小于等于这个的下标，再取前3个，就OK了

                if (num_real == 0):
                    temp_thresh = -1.0

                indexes_pruned = np.where(value1 <= temp_thresh)[
                    0]  # 因为value1是一维的话，这个返回的是(array([0,2,3], dtype=int64),)
                indexes_pruned = indexes_pruned[0:num_real]  # num_real所在的下标是不能取的
                # 如果用np.where取≥temp_thresh的话，不一定对，考虑好多个bn的通道数的权重一样的话， 这样有可能剪枝的和留下的重合了··
                # 所以应该从所有的index中拿出去上面的那些
                # 方法：把上面的值转化成list，然后，用range遍历索引，找出所有的值不在那个list中的数字，就收纳进去···很傻的方法···
                indexes_pruned = list(indexes_pruned)  # indexes_pruned = [0,3,6,7] (such as)
                indexes_retain = []

                for i in range(num_of_channel):
                    if (i not in indexes_pruned):
                        indexes_retain.append(i)
                # indexes_retain = [1,2,4,5]
                return indexes_pruned, indexes_retain


            elif (type(bns) is tuple):
                num_of_channel = self.bn_list[bns[0]].weight.data.size()[0]
                value1 = self.ratio_parabn * torch.abs(self.bn_list[bns[0]].weight.data).numpy() + (1-self.ratio_parabn) * torch.abs(
                    self.bn_list[bns[1]].weight.data).numpy()
                value2 = copy.deepcopy(value1)
                value2.sort()

                temp_thresh = value2[num_real - 1]
                if (num_real == 0):
                    temp_thresh = -1.0
                indexes_pruned = np.where(value1 <= temp_thresh)[0]
                indexes_pruned = indexes_pruned[0:num_real]

                indexes_pruned = list(indexes_pruned)
                indexes_retain = []

                for i in range(num_of_channel):
                    if (i not in indexes_pruned):
                        indexes_retain.append(i)
                return indexes_pruned, indexes_retain


        num_of_needed_pruned = [0 for i in self.bn_list]  #各个通道需要剪枝的数量
        index_of_pruned = [[] for i in self.bn_list]  #各个通道需要剪枝的通道的索引
        index_of_retain = [[] for i in self.bn_list]

        #预计返回上面的这两个列表，然后交给下一个函数去处理···因为感觉这个函数有点长了···不是和你好调bug了···


        if(version=="v1"):
            threshold = self.get_threshold_v1()
            self.threshhold = threshold
        elif(version=="v2"):
            threshold = self.get_threshold_v2(self.ratio_parabn)
            self.threshhold = threshold
        flag_bn_pruned = [False for x in self.bn_list]
        for i in range(len(self.bn_list)):  #这里一个个bn地剪枝！那些在dict包围中的东西，在底下发现根本没有上面constrained-bn
                                            #all-constrained-bn只有自己一个或者并行的两个。所以，你看，加不加dict括号，
                                            #这个剪枝的结构都是一样的（对于无并行module的模型来说）
            if(flag_bn_pruned[i]==True):
                continue
            flag_bn_pruned[i] = True  #说明这个通道已经剪枝
            #获取所有的依赖
            all_constrained_bn = []  #有平行的bn就放平行bn的元组
            num_of_needed_pruned_constrained = []

            current_bn = i  #这俩变量其实没必要···
            current_parallel_bn = None
            #先获取并列的bn
            if(current_bn in self.parallel_bn_dict.keys()):
                current_parallel_bn = self.parallel_bn_dict[current_bn]
            elif(current_bn in self.parallel_bn_list.keys()):
                current_parallel_bn = self.parallel_bn_list[current_bn]

            if(current_parallel_bn==None):
                temp_bns = current_bn
            else:
                temp_bns = (current_bn,current_parallel_bn)

            all_constrained_bn.append(temp_bns)
            while(temp_bns in self.constrained_bn.keys()):
                temp_bns = self.constrained_bn[temp_bns]
                all_constrained_bn.append(temp_bns)

            #把这些通道都变成True
            for i in all_constrained_bn:
                if(type(i) is int):
                    flag_bn_pruned[i] = True
                elif(type(i) is tuple):
                    flag_bn_pruned[i[0]] = True
                    flag_bn_pruned[i[1]] = True
            #获取剪枝i的所有的依赖完毕 在 all_constrained_bn 这个列表里面 已验证，没问题

            #all_constrained_bn = [17, 20, (23, 26)]   #没有通道限制，就是个[5] 或者 [(3,4)]

            #接下来衡量各个层需要剪枝的数量

            if (version == "v1"):
                for bns in all_constrained_bn:
                    num_of_needed_pruned_constrained.append(get_pruned_num_v1(threshold,bns))
            elif(version == "v2"):
                for bns in all_constrained_bn:
                    num_of_needed_pruned_constrained.append(get_pruned_num_v2(threshold,bns))\

            #num_of_needed_pruned_constrained 类似于 [(4,4),(5,5),3,(9,9),3,5,(1,1)]

            #现在感觉应该去找最小的剪枝数量，之后可以再调整···也好调整，反正得剪枝一样的数量
            num_real = get_pruned_num_of_all_constrained_bns_v1(num_of_needed_pruned_constrained)
            #现在num_real是真正的需要剪枝的数量！！！eg：3

            if(self.last_list_head != None):
                if(self.last_list_head in all_constrained_bn):
                    num_real = 0

            if(self.first_list_head != None):
                if(self.first_list_head in all_constrained_bn):
                    num_real = 0

            #num_of_needed_pruned = [0 for i in self.bn_list]  # 各个通道需要剪枝的数量
            #index_of_pruned = [[] for i in self.bn_list]  # 各个通道需要剪枝的通道的索引
            #index_of_retain = [[] for i in self.bn_list]
            #这三个是前面定义的几个变量
            for j in all_constrained_bn:  #有可能是int类型或者tuple类型
                if(type(j) is int):
                    temp_indexes_pruned,temp_indexes_retain = get_pruned_and_retain_index_v1(num_real,j)  #两个返回值都是
                                                                        #int类型的list
                    num_of_needed_pruned[j] = num_real
                    index_of_pruned[j] = temp_indexes_pruned
                    index_of_retain[j] = temp_indexes_retain
                elif(type(j) is tuple):
                    temp_indexes_pruned,temp_indexes_retain = get_pruned_and_retain_index_v1(num_real,j)

                    num_of_needed_pruned[j[0]] = num_real
                    num_of_needed_pruned[j[1]] = num_real
                    index_of_pruned[j[0]] = temp_indexes_pruned
                    index_of_pruned[j[1]] = temp_indexes_pruned
                    index_of_retain[j[0]] = temp_indexes_retain
                    index_of_retain[j[1]] = temp_indexes_retain


        return num_of_needed_pruned,index_of_pruned,index_of_retain
        #num_of_needed_pruned = [4,4,4,4,5,5,5,5]
        #index_of_pruned = [[0，1，2],[2，3，4],[5，7，8],[1，4，6],[6，7，8],[8，9，0]···]
        #index_of_retain = [[3,4,5],[0,1,5],[0,1,2,3,4],[0,2,3,5],[0,1,2,],[0,1,2,3]···]

        #注意 num_of_needed_pruned 和 num_of_needed_pruned_constrained 区别，后面这个是用在每个通道剪枝的时候的存储相互限制的哪几个
        #通道的需要剪枝的通道数 而前者是面对self.bn_list 整个model的bn的！



    def return_output_channel_convs(self):
        """
        格式按照conv-list来，只不过每个元素变成了输出通道
        :return:
        """
        output_channel_num = []
        for i in self.conv_list:
            output_channel_num.append(i.out_channels)

        return output_channel_num




    def return_last_list_head(self):
        """
        在获取了CFG-serial-number之后获取这个，就是说有可能CFG的最后一个元素是list，这样其实这个list开头的那个
        bn还是不能往前剪枝的，因为最后一层不能剪枝！但是如果是dict就还行···但是一般都是list
        如果不考虑这种情况，就会发生：constrained_bn：{5: 8, 8: (11, 14), 15: 18, 18: (21, 24), 25: 28}
        有28，但是28是不能剪枝的！所以需要处理这种情况

        方法：在获取每个通道需要剪枝的数量的时候，有一个获取所有通道限制的操作，在哪里，如果28（如果最后一个cfg是并行list，则是tuple）
        在那个列表中（或者字典？）则，所有相关联的剪枝通道数全变成0！
        :return:如果最后一个block不是list，则返回None，如果是list，则返回这个list的开头的bn序号！
        注意区分CFG[-1]是 30 还是 [30] 前者返回None，后者返回30
        """
        if type(self.CFG_serial_number[-1]) is list:
            if(type(self.CFG_serial_number[-1][0]) is int):
                return self.CFG_serial_number[-1][0]
            elif(type(self.CFG_serial_number[-1][0] is list)):
                return (self.CFG_serial_number[-1][0][0],self.CFG_serial_number[-1][1][0])

        return None

    def return_first_list_head(self):
        """
        和上面的函数向对应,但是，无论是什么类型，int dict list 都得返回啊！
        其实就是返回CFG的第一个元素序号，肯定有0，如果有并行，就返回两个这样子
        其实不用这么操作也行，因为0号对应的前驱conv是None
        :return:
        """
        if(type(self.CFG_serial_number[0]) is list):
            if(type(self.CFG_serial_number[0][0]) is int):
                return self.CFG_serial_number[0][0]
            elif(type(self.CFG_serial_number[0][0]) is list):
                return (self.CFG_serial_number[0][0][0],self.CFG_serial_number[0][1][0])
        elif(type(self.CFG_serial_number[0]) is int):
            return self.CFG_serial_number[0]

        elif(type(self.CFG_serial_number[0]) is dict):
            if(len(self.CFG_serial_number[0])==1):
                return self.CFG_serial_number[0][0][0]
            elif(len(self.CFG_serial_number[0])==2):
                return (self.CFG_serial_number[0][0][0],self.CFG_serial_number[0][1][0])

        return None







    def pruner_phase2(self):
        """
        通过三个新得到的量，计算出新的config
        需要用到bn-correlation-convs。
        遍历bn-list，取前驱conv，然后存到前驱序号对应的新的list中，表达新的输出通道
        然后把这个list再转化成config形式就行了

        bn_corralation_convs 是字典，拿到的是元组，元组第一个元素有可能有两个值。

        (1)先拿到类似于conv-list的东西
        (2)复制一个最初的cfg！包含avg，max的东西，然后遍历这个cfg！遍历的过程中保留current-num这个量，遇见avg，max就跳过
        就好像当初把通道数转换成序号的操作类似
        :return:
        """
        new_output_channel_num = [0 for i in self.conv_list]

        for i in range(len(self.num_of_needed_pruned)):  #self.num_of_needed_pruned = [4,4,4,4,5,5,5,5] 描述的是bn
            convs = self.bn_correlation_convs[i][0]  #bn-correlation-convs = (2,3) || ((3,4),5)
                                                     #convs = 2 || (3,4)
            if(type(convs) is int):
                new_output_channel_num[convs] = self.output_channel_num[convs] - self.num_of_needed_pruned[i]

            elif(type(convs) is tuple):
                new_output_channel_num[convs[0]] = self.output_channel_num[convs[0]] - self.num_of_needed_pruned[i]
                new_output_channel_num[convs[1]] = self.output_channel_num[convs[1]] - self.num_of_needed_pruned[i]

        #new_output_channel_num 计算完毕 [13,13,16,16,19,19,13,13,0]

        #计算到这里！new_output_channel_num有漏洞！因为最后一个conv，没有bn在他后面，所以在用convs做索引的时候，没有bn能
        #决定最后一个conv的输出通道！new_output_channel_num[convs[1]] 这种语句，索引根本就不会有最后一个conv。
        #所以new_output_channel_num[最后一个conv的序号] 等于初始化时候的值，就是0
        #所以，new_output。。。在那里就是0！！补救在new_model_config 基本完成之后补救··

        new_model_config = copy.deepcopy(self.CFG)

        current_num = 0
        for i in range(len(self.CFG)):
            if(type(self.CFG[i]) is str):
                continue
            elif(type(self.CFG[i]) is int):
                new_model_config[i] = new_output_channel_num[current_num]
                current_num += 1
            elif(type(self.CFG[i]) is list):
                if(type(self.CFG[i][0]) is int):  #[16,16,32]
                    for j in range(len(self.CFG[i])):
                        new_model_config[i][j] = new_output_channel_num[current_num]
                        current_num += 1
                elif(type(self.CFG[i][0]) is list):  # [[16,16,32],[32]]
                    for j in range(len(self.CFG[i][0])):
                        new_model_config[i][0][j] = new_output_channel_num[current_num]
                        current_num += 1

                    for j in range(len(self.CFG[i][1])):
                        new_model_config[i][1][j] = new_output_channel_num[current_num]
                        current_num += 1
            elif(type(self.CFG[i]) is dict):
                if(len(self.CFG[1])==1):  #{0:[16,16,32]}
                    for j in range(len(self.CFG[i][0])):
                        new_model_config[i][0][j] = new_output_channel_num[current_num]
                        current_num += 1
                elif(len(self.CFG[i])==2):  #{0:[16,16,32],1:[32]}
                    for j in range(len(self.CFG[i][0])):
                        new_model_config[i][0][j] = new_output_channel_num[current_num]
                        current_num += 1

                    for j in range(len(self.CFG[i][1])):
                        new_model_config[i][1][j] = new_output_channel_num[current_num]
                        current_num += 1
        assert current_num==len(self.conv_list)  #wrong!! current_num not == num of convs !!

        #补救最后一层的conv的输出通道
        #但是最后一层在CFG中有可能是avg之类的，所以要有一个不断往前推进的过程知道找到一个list，int，dict
        last_real_cfg_index = len(new_model_config)-1
        while(type(new_model_config[last_real_cfg_index]) is str):
            last_real_cfg_index -= 1

        if(type(new_model_config[last_real_cfg_index]) is int):
            new_model_config[last_real_cfg_index] = self.conv_list[-1].out_channels
        elif(type(new_model_config[last_real_cfg_index]) is list):
            if(type(new_model_config[last_real_cfg_index][0]) is int):
                new_model_config[last_real_cfg_index][-1] = self.conv_list[-1].out_channels
            elif(type(new_model_config[last_real_cfg_index][0]) is list):
                new_model_config[last_real_cfg_index][0][-1] = self.conv_list[-1].out_channels
                new_model_config[last_real_cfg_index][1][-1] = self.conv_list[-1].out_channels
        elif(type(new_model_config[last_real_cfg_index]) is dict):
            if(len(new_model_config[last_real_cfg_index])==1):
                new_model_config[last_real_cfg_index][0][-1] = self.conv_list[-1].out_channels
            elif(len(new_model_config[last_real_cfg_index])==2):
                new_model_config[last_real_cfg_index][0][-1] = self.conv_list[-1].out_channels
                new_model_config[last_real_cfg_index][1][-1] = self.conv_list[-1].out_channels

        return new_model_config



    def prune(self,version):
        self.num_of_needed_pruned,self.index_of_pruned,self.index_of_retain = self.pruner_phase1(version)
        self.new_config = self.pruner_phase2()

        #num_of_needed_pruned 是各个通道需要剪枝的数量
        #index_of_pruned 是各个通道要减去的索引
        #index_of_retain 是各个通道要留下的通道的索引

        return self.new_config


    def return_conv_correlation_bns(self):
        """

        条件：self.bn.correlation_convs

        从bn-correlation-convs应该可以直接导出这个需要计算的变量
        有一点需要注意的是，conv的前驱肯定就是这个conv的序号对应的bn，然后conv的后继不一定，但是肯定只有一个，不可能像
        bn的前驱一样有两个，然后求conv的后继的时候，应该从bn的前驱入手，遍历bn的前驱，找到对应的序号，然后这些conv的后就
        就是这个bn，但是在求新model的时候，似乎还得再求一遍？或者给新模型也加上个slim，然后自动就出来了，哈哈哈，不知道
        能不能容忍这种混乱。
        :return:
        """
        conv_correlation_bns = [[i,None] for i in range(len(self.bn_list))]  #这个就满足了前驱是序号的条件了

        for i in range(len(self.bn_correlation_convs)):
            if(type(self.bn_correlation_convs[i][0]) is int):  #self.bn_correlation_convs[i] = (2,3)
                conv_correlation_bns[self.bn_correlation_convs[i][0]][1] = i
            elif(type(self.bn_correlation_convs[i][0]) is tuple):  #self.bn_correlation_convs[i] = ((2,3),4)
                conv_correlation_bns[self.bn_correlation_convs[i][0][0]][1] = i
                conv_correlation_bns[self.bn_correlation_convs[i][0][1]][1] = i

        # 其实有点问题，就是有的conv对应两个bn的，但是没关系，我们只用其中一个bn就行了，因为这两个bn剪枝的情况是完全一样的···
        return conv_correlation_bns






def updateBN(model,s=0.0001):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))  # L1


