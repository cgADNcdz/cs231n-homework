#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:59:57 2018

@author: cdz
"""
import numpy as np
import SVM
import softmax
# ignore knn


############################# ********note****************************
# 1. in this project, the time to end depends on num_iters
# 2. We should destinguish SGD and BGD, in this project, we use SGD. Therefor, everytime we update the W(every interation),
# we use a batch of samples ranther than the whole X_train!!!!!!!
# 3.



class LinearClassifier(object):
    def __init__(self):
        self.W=None
        
    def train(self,X,Y,learning_rate=1e-3,regularization=1e-5,
              num_iters=100,batch_size=200,verbose=False):
        '''
        Train this linear classifier using stochastic gradient descent.
        Input:
            X: (N,D)
            Y:(N,)
            learning_rate: learning_rate
            regularization: regularizaiton strenth
            num_iters: number of steps to take when optimizing
            batch_size: number of training examples to use at ach step
            verbose: boolean, if true print progress during optinization
        Output:
            A list containing the value of loss function at each training iteration
        '''
        num_train,dim=X.shape
        num_classes=np.max(Y)+1
        if self.W is None:
            # lazily initialize W
            self.W=0.001*np.random.randn(dim,num_classes)
        
        # Run stochastic gradient descent to optimize W
        loss_history=[]
        for it in range(num_iters):
            X_batch=None
            Y_batch=None       # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是
            batch_idx=np.random.choice(num_train,batch_size,replace=False)
            X_batch=X[batch_idx]
            Y_batch=Y[batch_idx]
            
            # evaluate loss and gradient
            loss,grad=self.loss(X_batch,Y_batch,regularization)
            loss_history.append(loss)
        
            self.W=self.W-learning_rate*grad
        
            if verbose and it%100==0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            
        return loss_history
    
    
    def predict(self,X):
        '''
        use self.W to predict each sample of X
        input:
            X: an array of shape(N,D)
        output:
            y_pred:predicted labels(an integer) for each sample in X. The shape of y_pred is (N,)
        '''
        y_pred=np.zeros(X.shape[0])
        score_mat=X.dot(self.W)   # NxC
        y_pred=np.argmax(score_mat,axis=1)
        return y_pred
    
    def loss(self,X_batch,y_batch,regularization):
        '''
        compute the loss and dW
        return: (loss, grad)
        subclasses will override it
        '''
        
        

class LinearSVM(LinearClassifier):
           '''
           inherit from LinearClassifier
           '''
           
           def loss(self,X_batch,Y_batch,regularization):
               svm=SVM.SVM()
               return svm.svm_loss_vectorized(self.W,X_batch,Y_batch,regularization)
                  
        
    
class Softmax(LinearClassifier):
    def loss(self,X_batch,Y_batch,regularization):
        return softmax.softmax_loss_vectorized(self.W,X_batch,Y_batch,regularization)
    




























    