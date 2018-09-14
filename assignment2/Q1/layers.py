#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 22:39:19 2018

@author: cdz
"""

from builtins import range
import numpy as np


#################################  affine layer
def affine_forward(x,w,b):
    '''
    input:
        x: an array of shape(N,d_1,......d_k), which means N examples and each example has k dimensions
        w: an array of shape(D,M), where D is the dimension and M is the number of hidden units
        b: bias (M,) 
    retun: a tuple, (out,cache)
        out:output of shape(N,M)
        cache:(x,w,b)
    '''
    N=x.shape[0]
    '''
    D=1
    for i in range(1,len(x.shape)):
        D*=x.shape[i]
    '''
    D=int(np.prod(x.shape)/N) # prod return the product of an given array
    x_ND=x.copy()  # must use copy,otherwise, x will be changed
    x_ND.shape=(N,D) # N by d_1*d_2.....*d_k
    
    out=x_ND.dot(w)+b
    
    cache=(x,w,b)
    return out,cache


def affine_backward(dout,cache):
    '''
    input:
        dout: upstream derivative of shape(N,M)
        cache: x, input data of shape(N,d_1....d_k)
               w, weights of shape(D,M)
               b, bias of shape(M,)
    return: a tuple
         dx:Gradient with respect to x of shape(N,d_1.....d_k)
         dw:Gradient with respect to w of shape(D,M)
         db:Gradient with respect to b of shape(M,)            
    '''
    x,w,b=cache
    N=x.shape[0]
    x_ND=x.reshape(N,-1) # N by D
    #inplement the affine backward pass
    # out=wx+b
    # dx=αL/αout * aout/ax=dout * aout/ax
    # .......
    
    dx=dout.dot(w.T) # N by D
    dx=dx.reshape(*x.shape) # reshape to (N,d_1......d_k)
    
    dw=x_ND.T.dot(dout)
    
    db=np.sum(dout,axis=0)
    
    return dx,dw,db

    
def relu_forward(x):
    '''
    conpute the forward pass for a layer fo rectified linear units(relu)
    input:
        x:Inputs,of any shape
    return: a tuple
        out: output, of the same shape as x
        cache: x
    '''
    out=np.maximum(0,x)
    cache=x
    return out,cache
    

def relu_backward(dout,cache):
    '''
    conpute the backward pass for alayer of verctified linear units(ReLU)
    input:
        dout:upstream derivates, of any shape
        cache: input x,of the same shape as dout
    return:
        dx: gradient with respect to x
    '''           # maximum(0,x)
    x=cache # dx=aReLU/ax * dout;  f=wx+b(but here we use x to replace f)
    ax=np.int64(x>0)# derivate
    dx=dout*ax
 
    return dx
    
    
#######################  loss layer: softmax and svm(have written this in assignment1)

def svm_loss(x,y):
    '''
    compute the loss and gradient 
    input:
        x: of shape(N,C)
        y: labels
    return: a tuple
    loss: scalar
    dx:gradient of the loss with respect to x (x is actually the output of last layer)
    '''
    N=x.shape[0]
    correct_class_scores=x[np.arange(N),y]
    margins=np.maximum(0,x-correct_class_scores[:,np.newaxis]+1.0)
    margins[np.arange(N),y]=0
    loss=np.sum(margins)/N   # no regularization???????
    
    num_pos=np.sum(margins>0,axis=1)
    dx=np.zeros_like(x)
    dx[margins>0]=1
    dx[np.arange(N),y]-=num_pos # refer to the derivate of relu(x[y[i]] should be caculate num_pos times)
    dx/=N
    return loss,dx

def softmax_loss(x,y):
    '''
    ......omit
    '''                       # axis=1 means for every col (actually means for every col of each row)
    shifted_logits=x-np.max(x,axis=1,keepdims=True)
    Z=np.sum(np.exp(shifted_logits),axis=1,keepdims=True)  #∑
    log_probs=shifted_logits-np.log(Z)
    probs=np.exp(log_probs)
    N=x.shape[0]
    loss=-np.sum(log_probs[np.arange(N),y])/N # caculate just for every correct lasssification
    
    dx=probs.copy()
    dx[np.arange(N),y]-=1
    dx/=N
    return loss,dx

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    