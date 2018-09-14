#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 21:45:15 2018

@author: cdz
"""
import layers


def affine_relu_forward(x,w,b):
    '''
     Convenience layer that perorms an affine transform followed by a ReLU
     input:
         x:input to the affine layer
         w: wights
         b: bias
     return: a tuple
         out: output from the relu
         cache: object to give to the backward pass
    '''
    a,fc_cache=layers.affine_forward(x,w,b)  # a=wx+b fc_cache=(x,w,b)
    out,relu_cache=layers.relu_forward(a) # out=np.maximum(0,a)  relu_cache=a
    cache=(fc_cache,relu_cache)
    return out,cache

def affine_relu_backward(dout,cache):
    '''
    backward pass for the affine-relu convenience layer
    '''
    fc_cache,relu_cache=cache
    da=layers.relu_backward(dout,relu_cache)
    dx,dw,db=layers.affine_backward(da,fc_cache)
    return dx,dw,db
    
    
    















































