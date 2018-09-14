#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 19:07:39 2018

@author: cdz
"""
import numpy as np
import matplotlib.pyplot as plt


class Two_layer_net(object):
    '''
    a two-layer fully-connected neural network.
    input dimension:N
    hidden layer dimension:H
    class/output dimension: C
    softmax  Relu L2regulization
    '''
    
    def __init__(self,input_size,hidden_size,output_size,std=1e-4):
        '''
        input_size:D
        hidden_size: H
        output_size: C
        W1:First layer weight (D,H) ;    random initialize
        b1:First layer bias (H,)  ;      iniitalize by 0
        W2:Second layer weight (H,c) 
        b2:SEcond layer bias (C,)
        '''
        self.params={}
        self.params['W1']=std*np.random.randn(input_size,hidden_size)   
        self.params['b1']=np.zeros(hidden_size).reshape(-1,1).T
        self.params['W2']=std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size).reshape(-1,1).T
        
        
        
    def loss(self,X,y=None,regularization=0.0):
        '''
        X: an array of shape(N,D) X[i] is a training sample
        y: Vector of training labels, This parameter is optional;  If it is not passed then we only return scores,
           and if it is passed then we return the loss and gradient
        Return:
            If y is none, return a matrix socres of shape(N,C)
            If y is not none, return loss and gradient
        '''
        W1=self.params['W1']
        b1=self.params['b1']
        W2=self.params['W2']
        b2=self.params['b2']
        num_train=X.shape[0]
        
        
        # 1. forward pass,     N samples are caculated simultaneously
        scores=None
        f1=X.dot(W1)  # NxH
        h=np.maximum(0,f1+b1) # NxH   # ReLU,activate funtion; h is the output of hiddenlayer
        f2=h.dot(W2) #NxC
        scores=f2+b2 #NxC actually, scores are the output of the net
        
        if y is None:  # if y is none, just return the scores
            return scores
    
        # 2. compute the loss,  use softmax classifier loss!!!! 
        # and use L2 regularization for both W1,W2
        loss=0.0
        shift_scores=scores-np.max(scores,axis=1).reshape(-1,1)  # N by C
        softmax_output=np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape(-1,1) # N by C
        loss+=-np.sum(np.log(softmax_output[range(num_train),y]))            
        loss=loss/num_train
        loss+=0.5*regularization*np.sum(W1*W1)+0.5*regularization*np.sum(W2*W2)   # l2 eguarization
        
        # 3. backward pass: compute the gradients;  chain rule!!!!!
        grads={}
        d_scores=softmax_output.copy()  # d_scores is the gradient on scores (actually it is  αL/αf, where f is the scores,f=wx)
        d_scores[range(num_train),y]-=1  # aL/af, where f=h*w2+b2
        d_scores/=num_train # N by C (the reason why /num_train is that loss must divide num_train, and d_scores is the derivative of loss)
        #first backpropgate into parameters W2 and b2
        dW2=np.dot(h.T,d_scores)  # chain rule, aL/af *af/aw2  ,H by C
        db2=np.sum(d_scores,axis=0,keepdims=True)  # (C,), af/ab2=1
        # next backprop into hidden layer
        dh=np.dot(d_scores,W2.T) # aL/ah=aL/af *af/ah
        # then backpropgate the ReLU non-linearity
        dh[h<=0]=0  # max(0,w1x+b1)=> ah/a(w1+b1)=0 or (w1x+b1), after the operation, dh is aL/a(w1+b1)  
        #finally backpropgate into W1, b1
        dW1=np.dot(X.T,dh)
        db1=np.sum(dh,axis=0,keepdims=True)
        
        dW2+=regularization*W2  # b is no need to be regularized
        dW1+=regularization*W1
        grads['W1']=dW1   # store in grads and return 
        grads['b1']=db1
        grads['W2']=dW2
        grads['b2']=db2
        
        return loss,grads
    
    
    
    
    
    
    
    def train(self,X,y,X_val,y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iterations=100,
              batch_size=200, verbose=False):
        '''
        using stochastic gradient descent
        X: (N,D)
        Y: (n,)
        X_val: validation data,(N_val,D)
        y_val: validation label,(N_val,)
        ......
        batch_size: number of training examples to use per step
        verbose: boolean; if true print progress during optimization
        '''
        num_train=X.shape[0]
        iterations_per_epoch=max(num_train/batch_size,1)
        
        # use SGD to optimize the parameters 
        loss_history=[]
        train_acc_history=[]
        val_acc_history=[]
        
        # each iteration use a mini batch
        for i in range(num_iterations):
            index=np.random.choice(num_train,batch_size,replace=True)
            X_batch=X[index]   
            y_batch=y[index]
            
            # compute the gradients and loss using current minibatch
            loss,grads=self.loss(X_batch,y=y_batch,regularization=reg)
            loss_history.append(loss)
            
            # update the parameters 
            self.params['W2']+=-learning_rate*grads['W2']
            self.params['b2']+=-learning_rate*grads['b2']
            self.params['W1']+=-learning_rate*grads['W1']
            self.params['b1']+=-learning_rate*grads['b1']
            
            if verbose and i%100==0:
                print('iteration %d/%d: loss %f'%(i,num_iterations,loss))
            
            # for each mini batch
            if i%iterations_per_epoch==0:
                train_acc=(self.predict(X_batch)==y_batch).mean()
                val_acc=(self.predict(X_val)==y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                # decay learning rate
                learning_rate*=learning_rate_decay
            
        return {
                'loss_history':loss_history,
                'train_acc_history':train_acc_history,
                'val_acc_history':val_acc_history
                }
                
    
    def predict(self, X):
        W1=self.params['W1']
        b1=self.params['b1']
        W2=self.params['W2']
        b2=self.params['b2']
        
        f1=X.dot(W1)
        h=np.maximum(0,f1+b1)
        f2=h.dot(W2)
        scores=f2+b2
        
        y_pred=np.argmax(scores,axis=1)
        
        return y_pred
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        