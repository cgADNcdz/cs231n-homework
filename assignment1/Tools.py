#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:15:12 2018

@author: cdz
"""
import pickle
import os
import numpy as np
from random import randrange





        
#get photo data(a dictionary [data,label] data:10000x3072) 
def unpickle(self,file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
#get the batch_meta(return a dictionary too)
def get_meta(self,file):
    with open(file,'rb') as fo:
        meta_dict=pickle.load(fo,encoding='bytes')
    return meta_dict
    

#######********************************************************   these are changed from cs231n

#################################3   about get data
#get X_train,Y_trian ,x_test,y_test   (cubic!!!)
    
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='bytes')
    X = datadict[b'data']
    Y = datadict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT):
    ''' load all cifar'''
    x_temp=[]
    y_temp=[]
    for i in range(1,6):
        X,Y=load_CIFAR_batch(os.path.join(ROOT,'data_batch_%d'%(i,)))
        x_temp.append(X)
        y_temp.append(Y)
    X_train=np.concatenate(x_temp)
    Y_train=np.concatenate(y_temp)
    
    del X,Y
    
    X_test,Y_test=load_CIFAR_batch(os.path.join(ROOT,'test_batch'))
    
    return X_train,Y_train,X_test,Y_test

    
        
        
    
##########################  about gradient

# check some of the x  
def grad_check_sparse(f,x,analytic_grad,num_checks=10,h=1e-5):
    '''
    f: function
    sample a few random elements and only return numerical in this dimension
    '''
    for i in range(num_checks):   
        ix=tuple([randrange(m) for m in x.shape])   # just choose some of the gradient, not compute all!!!!!!!!
        
        oldval=x[ix]
        
        x[ix]=oldval+h  # increment by h
        fxph=f(x)   #compute f(x+h)
        
        x[ix]=oldval-h  #increment by -h
        fxmh=f(x)   #conpute f(x-h)
        
        x[ix]=oldval  # reset
        
        grad_numerical=(fxph-fxmh)/(2*h)
        grad_analytic=analytic_grad[ix]
        
        rel_error=abs(grad_numerical-grad_analytic)/(abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

# check all in x
def eval_numerical_gradient(f,x,verbose=True,h=0.00001):
    '''
    '''
    fx=f(x) # evaluate function value at original point
    grad=np.zeros(x.shape)
    
    # iterate over all index in x
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        index=it.multi_index
        old_val=x[index]
        
        x[index]=old_val+h #increment by h
        fxph=f(x)
        
        x[index]=old_val-h
        fxmh=f(f)
        
        x[index]=old_val # restore
        
        # compute the partial derivative with centered formula
        grad[index]=(fxph-fxmh)/(2*h)
        
        if verbose:
            print(index,grad[index])
        it.iternext() # step to next dimension
    return grad
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    