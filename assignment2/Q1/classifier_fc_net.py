#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:31:34 2018

@author: cdz
"""

from builtins import range
from builtins import object
import numpy as np
import layers
import layer_utilities


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
    def __init__(self,input_dim=3*32*32,hidden_dim=100,num_classes=10,
                 weight_scale=1e-3,reg=0.0):
        
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params={}
        self.reg=reg
        
        self.params['W1']=weight_scale*np.random.randn(input_dim,hidden_dim)
        self.params['b1']=np.zeros(hidden_dim)
        self.params['W2']=weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b2']=np.zeros(num_classes)
    
    def loss(self,X,y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        out1,cache1=layer_utilities.affine_relu_forward(X,self.params['W1'],self.params['b1'])
        out2,cache2=layers.affine_forward(out1,self.params['W2'],self.params['b2'])  # last layer no need to use relu
        scores=out2
      
        if y is None:
            return scores
        # backward  
        loss,grads=0,{}
        loss,d_scores=layers.softmax_loss(scores,y)
        loss=loss+0.5*self.reg*(np.sum(self.params['W1']*self.params['W1'])+np.sum(self.params['W2']*self.params['W2']))
      
        dout1,dW2,db2=layers.affine_backward(d_scores,cache2)
        dx,dW1,db1=layer_utilities.affine_relu_backward(dout1,cache1)
      
        grads['W2']=dW2+self.reg*self.params['W2']
        grads['b2']=db2
        grads['W1']=dW1+self.reg*self.params['W1']
        grads['b1']=db1
      
        return loss,grads
    
    
    
#class FullyConnectedNet(object):
    
      
      
  
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
