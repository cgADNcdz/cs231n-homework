#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:06:34 2018

@author: cdz
"""
import numpy as np
def softmax_loss_naive(W,X,y,regularization):
    '''
    using loops
    W: weights (D,C)
    X: samples (N,D)
    y: labels of X (N,)
    regularization: regularization strength
    
    output:
        (loss grad)
    '''
    loss=0.0
    dW=np.zeros(W.shape)
    
    num_train=X.shape[0]
    num_classes=W.shape[1]    
    scores=X.dot(W)  # NxC
    
    for i in range(num_train):
        scores_i=scores[i,:]
        shift_scores=scores_i-np.max(scores_i)  # in numpy log means ln we use; log10 is real log we use usually                                
        loss+=-shift_scores[y[i]]+np.log(sum(np.exp(shift_scores)))
        for j in range(num_classes):
            softmax_output=np.exp(shift_scores[j])/sum(np.exp(shift_scores))
            if j==y[i]:            # i can't really understand how to caculate the gradient here
                dW[:,j]+=(-1+softmax_output)*X[i]   
            else:
                dW[:,j]+=softmax_output*X[i]
                
    loss/=num_train
    loss+=0.5*regularization*np.sum(W*W)
    dW/=num_train
    dW+=regularization*W
    
    return loss,dW


            
def softmax_loss_vectorized(W,X,y,regularization):
        loss=0.0
        dW=np.zeros(W.shape)
        
        # you had better refer to: http://cs231n.github.io/neural-networks-case-study/
        num_train=X.shape[0]
        num_classes=W.shape[1]
        scores=X.dot(W)  # NxC
        shift_scores=scores-np.max(scores,axis=1).reshape(-1,1)  #broadcast  NxC
        softmax_output=np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)  # NxC
        loss=-np.sum(np.log(softmax_output[range(num_train),y]))   # attention: just caculate every correct samples
        loss/=num_train
        loss+=0.5*regularization*np.sum(W*W)
        
        # interpretion aout the following: 
        # actually, dS is the gradient on scores(αL/αf, f is scores,f=wx)
        # then,we can use chain rule to caculate dW, αL/αW=αL/αf *αf/αW
        dS=softmax_output.copy()
        dS[range(num_train),y]+=-1   # where j==y[i]
        dW=X.T.dot(dS)
        dW=dW/num_train+regularization*W
        
        return loss,dW
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    