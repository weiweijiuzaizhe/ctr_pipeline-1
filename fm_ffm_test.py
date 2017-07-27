#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import  random
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class fm(object):
    '''
    fm class is implementation for fm algorithm, and use Least squares as optimizer, the train data wasn't any encoded

    Attributes:
        datafile: the data source path will be used to train and valify
        data_x_dim: the number of data rows
        onehot_model： True: kv encode for data,ex. low1: 255:1,512:1
    '''
    '''
    说明：fm_oneHot 类初始化
    @datafile：the path of datafile
    @data_x_dim:总行数
    @data_y_dim:总列数
    '''
    def __init__(self,data):
        self.train_x_data = data[:,1:]
        self.train_y_data = data[:,0]
        self.val_x_data = self.train_x_data
        self.val_y_data = self.train_y_data

    '''
    说明：cal_y计算y预估值，y= w0 + Σ(wi*xi) +ΣΣ<vi,vj>xixj
    @param model: 模型参数
    @param index：训练数据索引值，代表第几条
    @return: 返回计算后的预估值
    '''
    def cal_y(self,model,index):
        data = self.train_x_data[index]
        W, W0, V, lambd = model['W'], model['W0'], model['V'], model['lambd']
        y = W0 + np.sum(W * data)#y= w0 + Σ(wi*xi)
        sum_xij = 0
        for i in range(len(data) - 1):
            for j in np.arange(i + 1, len(data), 1):
                if data[i] != 0 and data[j] != 0:
                    # sum_xij += dd[i] * dd[j] * (V.T[i].dot(V.T[j]))
                    d_valuer = data[i] * data[j]
                    v_valuer = np.sum(V[:, i] * V[:, j])

                    sum_xij += d_valuer*v_valuer


        return y + sum_xij#y= w0 + Σ(wi*xi) +ΣΣ<vi,vj>xixj

    '''
   说明：对V向量进行求偏导 dv =xi∑vj,f*xj -vi,f xi²
   @param model: 模型参数
   @param lineData：单条数据
   @return: 返回计算后的V向量
   '''
    def cal_dv(self,model, lineData):
        V,lambd= model['V'],model['lambd']
        dv = np.ones_like(V)
        for f in range(len(V)):
            sum_vjxj = np.sum(V[f] * lineData)
            for i in xrange(len(lineData)):
                dv[f][i] = lineData[i] * sum_vjxj - V[f][i] * np.square(lineData[i])
                if dv[f][i]>0:
                    dv[f][i] = dv[f][i] + lambd
                else:
                    dv[f][i] = dv[f][i] - lambd
        return dv

    '''
    说明：目标函数 Loss = 1/2 * [y(x)-y]² +λ*Σ|W|
    @param model: 模型参数
    @param index：训练数据索引值，代表第几条
    @return: 返回计算后的Loss
    '''
    def calculate_loss(self,model, index):
        W, W0, V, lambd = model['W'], model['W0'], model['V'],model['lambd']
        return np.square(self.train_y_data[index] - self.cal_y(model, index)) * .5 +sum(abs(W)*lambd)

    '''
    说明：模型训练
    @param dim_k: V向量长度f值
    @param num_passes：迭代次数
    @param print_loss：是否打印出损失值，True:是，False:否
    @return: 返回完成训练的模型
    '''
    def build_model(self, lr,dim_k, num_passes=20000, print_loss=False):
        '''
        参数：
        1) dim_k :   V向量长度
        2）num_passes: 梯度下降迭代次数
        3）print_loss: 设定为True的话，每1000次迭代输出一次loss的当前值
        '''
        # 随机初始化一下权重
        dim_data = len(self.train_x_data[0])
        W = np.random.randn(dim_data) / np.sqrt(dim_data)
        V = np.random.randn(dim_k, dim_data) / np.sqrt(dim_data)
        W0 = 1
        lambd = 0
        learnRate = lr

        print "\nW:", W, "\nV:", V, "\nW0:", W0
        model = {'W': W, 'W0': W0, 'V': V,'lambd':lambd}
        old_loss = 1
        # 开始梯度下降...
        for i in range(num_passes):

            data_index= i%len(self.train_x_data)
            iterData = self.train_x_data[data_index]
            y_yx = self.cal_y(model, data_index) - self.train_y_data[data_index]
            # print "\n ----------cal_y:",self.cal_y(model, data_index),"   y:",self.train_y_data[data_index]

            # print "\ny_yx:",y_yx
            # dW0 = y_yx * W0
            if W0>0 :
                dW0 = y_yx * W0 * (1 + lambd)
            else:
                dW0 = y_yx * W0 * (1 - lambd)


            # dW = y_yx * iterData
            dW =[]
            for j in iterData:
                if j >0:
                    dW.append(y_yx *(j+lambd))
                else:
                    dW.append(y_yx *(j -lambd))

            dV = y_yx * self.cal_dv(model, iterData)


            # updata weights
            W0 -= dW0 * learnRate
            V -= dV * learnRate
            W -= np.array(dW) * learnRate
            model = {'W': W, 'W0': W0, 'V': V, 'lambd': lambd}

            # print "-------------------\nmodel:",model
            if print_loss and i % 100 == 0:
                loss = self.calculate_loss(model, data_index)
                pre_y = self.cal_y(model, data_index)
                # print "\n model:",model
                print "===========Loss after iteration %i: %f y=%i pre=%f" % ( i, loss, self.train_y_data[data_index], pre_y)



        return model

    '''
    说明：获取one hot编码的训练数据，并对其进行解码，还原成矩阵形式
    @param data_path: 数据文件路径
    @x_dim：行数
    @ydim:列数
    @return: 训练数据集
    '''
    def get_train_x_data_decodeOneHot(self,data_path,x_dim,ydim):
        x = np.zeros((x_dim, ydim))
        for i, d in enumerate(file(path)):
            d = d.strip()
            if not d:
                continue
            d = map(str, d.split(' '))[1:]
            for data in d:
                x[i][int(data.split(":")[0])] = 1
        return x

    '''
    说明：获取one hot编码的训练数据，并对其进行解码，还原成矩阵形式
    @param data_path: 数据文件路径
    @return: y标签数据
    '''
    def get_train_y_data_decodeOneHot(self,data_path):
        y=[]

        for i, d in enumerate(file(data_path)):
            d = d.strip()
            if not d:
                continue
            d = map(str, d.split(' '))[0]
            y.append(d)
        return np.array(y).astype(dtype=np.int)



class fm_oneHot(object):
    '''
    fm class is implementation for fm algorithm, and use Least squares as optimizer, the train data was encoded by onehot

    Attributes:
        datafile: the data source path will be used to train and valify
        data_x_dim: the number of data rows
        onehot_model： True: kv encode for data,ex. low1: 255:1,512:1
    '''

    '''
    说明：fm_oneHot 类初始化
    @datafile：the path of datafile
    '''
    def __init__(self,datafile ):
        self.train_x_data = self.get_train_x_data(datafile)
        self.train_y_data = self.get_train_y_data(datafile)
        self.val_x_data = self.train_x_data
        self.val_y_data = self.train_y_data

    '''
    说明：cal_y计算y预估值，y= w0 + Σ(wi*xi) +ΣΣ<vi,vj>xixj
    @param model: 模型参数
    @param index：训练数据索引值，代表第几条
    @return: 返回计算后的预估值
    '''
    def cal_y(self,model,index):
        data = self.train_x_data[index]
        W, W0, V  = model['W'], model['W0'], model['V']
        y = W0 + np.sum(W[data])  # y= w0 + Σ(wi*xi)
        sum_y_v = 0 #sum_y_v=ΣΣ<vi,vj>xixj
        for i in range(len(data) - 1):
            for j in np.arange(i + 1, len(data), 1):
                sum_y_v += np.sum(V[:, data[i]] * V[:, data[j]])#计算<vif,vjf>

        return y + sum_y_v  #y= w0 + Σ(wi*xi) +ΣΣ<vi,vj>xixj

    '''
    说明：对V向量进行求偏导 dv =xi∑vj,f*xj -vi,f xi²
    @param model: 模型参数
    @param lineData：单条数据
    @return: 返回计算后的V向量
    '''
    def cal_dv(self,model, lineData):
        V,lambd= model['V'],model['lambd']

        for f in range(len(V)):
            sum_vjxj = np.sum(V[f][lineData])#
            for i,data in enumerate(lineData):
                V[f][data] =  sum_vjxj - V[f][data]#xi=1
                if V[f][data]>0:
                    V[f][data] = V[f][data] + lambd
                else:
                    V[f][data] = V[f][data] - lambd
        return V

    '''
    说明：目标函数 Loss = 1/2 * [y(x)-y]² +λ*Σ|W|
    @param model: 模型参数
    @param index：训练数据索引值，代表第几条
    @return: 返回计算后的Loss
    '''
    def calculate_loss(self,model, index):
        W, W0, V, lambd = model['W'], model['W0'], model['V'],model['lambd']
        return np.square(self.train_y_data[index] - self.cal_y(model, index)) * .5 +lambd*(np.sum(abs(W))+abs(W0)+np.sum(abs(V)))

    '''
    说明：模型训练
    @param dim_k: V向量长度f值
    @param num_passes：迭代次数
    @param print_loss：是否打印出损失值，True:是，False:否
    @return: 返回完成训练的模型
    '''
    def build_model(self,dim_k, num_passes=20000, print_loss=False):
        '''
        参数：
        1) dim_k :   V向量长度
        2）num_passes: 梯度下降迭代次数
        3）print_loss: 设定为True的话，每1000次迭代输出一次loss的当前值
        '''
        # 随机初始化一下权重
        dim_para =0
        for iter_data in self.train_x_data:
            length= max(iter_data)
            if length>dim_para:
                dim_para = length
        print "dim_para:",dim_para
        W = np.random.randn(dim_para) / np.sqrt(dim_para)
        # V = np.random.randn(dim_k, dim_para) / np.sqrt(dim_para)
        V = np.full((dim_k,dim_para),0)
        W0 = 1
        lambd = 0
        learnRate = .01

        print "len(W):",len(W)
        print "len(V):",len(V)
        print "len(V[0]):", len(V[0])
        print "\nW:", W, "\nV:", V, "\nW0:", W0
        print "V:",V
        model = {'W': W, 'W0': W0, 'V': V,'lambd':lambd}
        old_loss = 1
        # 开始梯度下降...
        print "start sgd..."
        for i in range(num_passes):

            data_index= i%len(self.train_x_data)
            iterData = self.train_x_data[data_index]
            y_yx = self.cal_y(model, data_index) - self.train_y_data[data_index]
            # print "\n ----------cal_y:",self.cal_y(model, data_index),"   y:",self.train_y_data[data_index]



            # print "\ny_yx:",y_yx
            dW0 = y_yx * W0
            if W0>0 :
                dW0 += lambd
            else:
                dW0 -= lambd



            dW =np.zeros_like(W)

            for iter in iterData:
                if W[iter]>0:
                    dW[iter]= y_yx + lambd
                else:
                    dW[iter] = y_yx - lambd


            dV = y_yx * self.cal_dv(model, iterData)
            sum_para = np.square(dW0) + np.sum(np.square(dW)) + np.sum(np.square(dV))
            print "-------sum_para:",sum_para
            # updata weights
            W0 = W0 - dW0 * learnRate/np.sqrt(sum_para)
            V = V - dV * learnRate/np.sqrt(sum_para)
            W = W - dW * learnRate/np.sqrt(sum_para)
            model = {'W': W, 'W0': W0, 'V': V, 'lambd': lambd}

            # print "-------------------\nmodel:",model
            # if print_loss and i % 100 == 0:

            if print_loss :
                current_loss = self.calculate_loss(model, data_index)
                pre_y = self.cal_y(model, data_index)
                # print "\n model:",model
                print "===========Loss after iteration %i: %f y=%i pre=%f" % (i, current_loss,self.train_y_data[data_index],pre_y)


                if abs(current_loss-old_loss)/current_loss <= 0.05:
                    # break
                    pass;
                old_loss = current_loss

        return model

    '''
    说明：获取one hot编码的训练数据
    @param data_path: 数据文件路径
    @return: 训练数据集
    '''
    def get_train_x_data(self,data_path):
        x = []
        for i, d in enumerate(file(data_path)):
            d = d.strip()
            if not d:
                continue
            d = map(str, d.split(' '))[1:] # 0 为y, [1:]为xi
            d = [map(str, i.split(':'))[0] for i in d]  # 切片254：1 只保留254，即每行数据保留均为1的index,未保留的均为0
            d = [int(i) for i in d]#convert string type to int
            x.append(d)
        return np.array(x)

    '''
    说明：获取one hot编码的训练数据
    @param data_path: 数据文件路径
    @return: 训练数据集
    '''
    def get_train_y_data(self,data_path):
        y=[]
        for i, d in enumerate(file(data_path)):
            d = d.strip()
            if not d:
                continue
            d = map(str, d.split(' '))[0]
            y.append(d)
        return np.array(y).astype(dtype=np.int)

def test_fmonehot():
    path = '/Users/bruce/Documents/one_hot_criteo_10w.txt'
    train = fm_oneHot(path)
    train.build_model(1,3001,True)
def test_fm():
    k_dim=1
    featureNum=100
    dataRows=10
    data = np.random.randint(0,2,(dataRows,featureNum+1))#col 0:y, col1~featureNum+1:x
    train = fm(data)
    lr=0.001
    train.build_model(lr,k_dim,2001,True)

if __name__=='__main__':
    # test_fmonehot()
    test_fm()
