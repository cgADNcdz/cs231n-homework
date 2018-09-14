#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:01:56 2018

@author: cdz
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import Tools

class SVM(object):
    
    def __init__(self):
        pass

    def svm_loss_naive(self,W,X,y,regularization):
        '''
        Inputs:(D:dimension  N:number of samples C:number of labels(classes))
             W: A numpy array of shape(D,C) containing weights
             X: A numpy array of shape(N,D) containing a minibatch of data
             y: A numpy array of shape(N,) contaning training labels, y[i]=c, means that X[i] has the label c
             regularization: regularization strength
        Return: (a tumple)
             loss: 
             dW:
        Attention: compute dWj and dYi is really hard to fully undersatand, be careful!!!!!!
        '''
        dW=np.zeros(W.shape)
        
        #conpute the loss and the gradient
        num_classes=W.shape[1]
        num_train=X.shape[0]
        loss=0.0
        for i in range(num_train):
            scores=X[i].dot(W)  #1xC scores[m] is the score of X[i] in mth class
            correct_class_score=scores[y[i]]  # X[i]'s score in corrrct sample 
            for j in range(num_classes):
                if j==y[i]: # when j==yi,we don't caculate loss
                    continue
                margin=scores[j]-correct_class_score+1  # note delta=1 
                if margin>0:     # Li=∑j≠yi max(0,Sj-yi+delta)
                    loss+=margin
                    dW[:,y[i]]+=-X[i].T  # αLi/αWyi      ∇wyiLi=−(∑j≠yi𝟙(wTjxi−wTyixi+Δ>0))xi
                    dW[:,j]+=X[i].T  # αLi/αWj       refer to:  http://cs231n.github.io/optimization-1/                             
        loss/=num_train              # dWj=αL1/αWj+ αL2/αWj+ αL2/αWj.....     Li是一个样本的总损失，dWj是损失对一个类的判别参数求导的结果
        dW/=num_train                # margin≤0时，求导结果为0，故没有加上去
        
        # add regularization to the loss
        loss+=0.5*regularization*np.sum(W*W)
        dW+=regularization*W
    
        return loss,dW
    
    
    def svm_loss_vectorized(self,W,X,y,regularization):
        '''
        use vetor rather than lops
        
        Attention: it is really hard to understand the vectorization here, be patient!!!!
        '''
        loss=0.0
        dW=np.zeros(W.shape)
        
        
        num_train=X.shape[0]
        num_classes=W.shape[1]
        scores=X.dot(W)  # NxC     range: from 0 to num_train;   become Nx1
        correct_class_scores=scores[range(num_train),y].reshape(-1,1)
        margin=np.maximum(0,scores-correct_class_scores+1)   #  NxC      broadcast
        margin[range(num_train),y]=0   # where the ith class is the correct class(j==y[i]), this place don't caculate loss
        
        loss+=np.sum(margin)/num_train+0.5*regularization*np.sum(W*W)
      
        #  why?????????
        coeff_mat=np.zeros((num_train,num_classes))
        coeff_mat[margin>0]=1     #置1的目的是dWj这一项，让所有样本的Xi相加;而margin<0时不计入loss，所以置0
        coeff_mat[range(num_train),y]=0    #将j==y[i]的项=0，因为求loss function时要求j!=y[i],所以这一项置0
        coeff_mat[range(num_train),y]=-np.sum(coeff_mat,axis=1)   #等号后面是(N,);dWyi要计算C-1次，因为每一次计算dWj时都计算了一次dWyi(line46)
                                                                  # dWj=X.T.dot(coeff_mat[:,j])
        dW=X.T.dot(coeff_mat)                                     # dWyi=-X.T.dot(coeff_mat[:,y[i])
        dW=dW/num_train+regularization*W

        return loss,dW


































