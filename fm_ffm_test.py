#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import  random
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class fm(object):
    def __init__(self,datafile,data_x_dim,data_y_dim):
        self.train_x_data = self.get_train_x_data(datafile,data_x_dim,data_y_dim)
        self.train_y_data = self.get_train_y_data(datafile)
        self.val_x_data = self.train_x_data
        self.val_y_data = self.train_y_data

    #预测
    def cal_y(self,model,index):
        dd = self.train_x_data[index]
        W, W0, V, lambd = model['W'], model['W0'], model['V'], model['lambd']
        y = W0 + np.sum(W * dd)
        sum_xij = 0
        # print "\n\nV:",V

        for i in range(len(dd) - 1):
            for j in np.arange(i + 1, len(dd), 1):
                if dd[i] != 0 and dd[j] != 0:
                    # sum_xij += dd[i] * dd[j] * (V.T[i].dot(V.T[j]))
                    d_valuer = dd[i] * dd[j]
                    v_valuer = np.sum(V[:, i] * V[:, j])

                    sum_xij += d_valuer*v_valuer

        return y + sum_xij


    #对V向量进行求偏导
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

        # dv = np.ones_like(v)
        # sum123 = 0
        # for i in range(len(v)):
        #     sum123 += np.sum(v[i] * lineData)
        #
        # for i in range(len(v)):
        #     for ii in xrange(len(lineData)):
        #         dv[i][ii] = lineData[ii] * sum123 - v[i][ii] * np.square(lineData[ii])
        # return dv

    #目标函数最小二乘法
    def calculate_loss(self,model, index):
        W, W0, V, lambd = model['W'], model['W0'], model['V'],model['lambd']
        return np.square(self.train_y_data[index] - self.cal_y(model, index)) * .5 +sum(abs(W)*lambd)


    #训练模型
    def build_model(self, dim_k, num_passes=20000, print_loss=False):
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
        learnRate = .01

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

            sum_para =W0**2+np.sum(np.square(dW))+np.sum(np.square(dV))
            sum_para =1

            # updata weights
            W0 -= dW0 * learnRate/np.sqrt(sum_para)
            V -= dV * learnRate/np.sqrt(sum_para)
            W -= np.array(dW) * learnRate/np.sqrt(sum_para)
            model = {'W': W, 'W0': W0, 'V': V, 'lambd': lambd}

            # print "-------------------\nmodel:",model
            # if print_loss and i % 100 == 0:

            if print_loss :
                current_loss = self.calculate_loss(model, data_index)

                # print "\n model:",model
                print "===========Loss after iteration %i: %f" % (i, current_loss)

                if abs(current_loss-old_loss)/current_loss <= 0.05:
                    # break
                    pass;
                old_loss = current_loss

        return model

    #获取计算结果的上三角
    def getEpitriquetrum(self,matrix):
        if type(matrix) != np.ndarray:
            raise Exception("data type error")
        b = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i + 1 + j < len(matrix[0]):
                    b.append(matrix[i][i + 1 + j])
        return b

    #trainData ETL
    def calTrainData(self,matrix):
        if type(matrix) != np.ndarray:
            raise Exception("data type error")
        data = []
        for i in range(len(matrix)):
            data.append(self.getEpitriquetrum(matrix[i] * matrix[i].reshape(-1, 1)))

        return np.c_[matrix, data]
    #从one hot 编码数据获得训练数据
    def get_train_x_data(self,data_path,x_dim,ydim):
        x = np.zeros((x_dim, ydim))
        for i, d in enumerate(file(path)):
            d = d.strip()
            if not d:
                continue
            d = map(str, d.split(' '))[1:]
            for data in d:
                x[i][int(data.split(":")[0])] = 1
        return x

    # 从one hot 编码数据获得训练数据
    def get_train_y_data(self,data_path):
        y=[]

        for i, d in enumerate(file(data_path)):
            d = d.strip()
            if not d:
                continue
            d = map(str, d.split(' '))[0]
            y.append(d)
        return np.array(y).astype(dtype=np.int)

if __name__=='__main__':
    '''样本数据 Y,X1,X2'''
    # Data = np.array([
    #     [1, 0, 1],
    #     [0, 1, 1],
    #     [1, 0, 0],print "\n\n\n\n"
    #     [1, 1, 0]
    # ])

    path = '/Users/bruce/Documents/one_hot_criteo_10w.txt'
    x_axis = 100000
    y_axis = 8000000
    train =   fm(path,x_axis,y_axis)
    print "xdata[0][255]:",train.train_x_data[0][255],"xdata[0][256]:",train.train_x_data[0][256]
    # Data = np.random.randint(0,2,(3000,10))
    # print "\nData:",Data
    #
    #
    # X = Data[:, :-1]
    # Y = Data[:, -1]
    # #
    # train = fm(X,Y,X,Y)
    train.build_model(10,3001,True)



