#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 13:30:59 2018

@author: cdz
"""

import random
import numpy as np
import Tools
import softmax
import matplotlib.pyplot as plt
import time
import liner_classifiers_total

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def get_CIFAR10_data(num_training=49000,num_validation=1000,
                     num_test=1000,num_dev=500
                     ):
    '''
    actually it is the same with what i have wrote in SVM_test,
    but here,i write them in a function
    '''
    # load raw CIFAR10 data
    X_train,Y_train,X_test,Y_test=Tools.load_CIFAR10('cifar-10-batches-py')
    
    # subset the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    Y_dev = Y_train[mask]
    
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev



# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
    del X_train,Y_train
    del X_test,Y_test
    print('clear previously loaded data')
except:
    pass

# invoke the above function to get our data
X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev=get_CIFAR10_data()
print('X_train shape: '+str(X_train.shape))
print('Y_train shape: '+str(Y_train.shape))
print('X_test shape: '+str(X_test.shape))
print('Y_test shape: '+str(Y_test.shape))
print('X_val shape: '+str(X_val.shape))
print('Y_val shape: '+str(Y_val.shape))
print('X_dev shape: '+str(X_dev.shape))
print('Y_dev shape: '+str(Y_dev.shape))


#############################################################################
#################################  softmax classifier

# First implement the naive softmax loss function with nested loops.

W=np.random.randn(3073,10)*0.0001
loss,grad=softmax.softmax_loss_naive(W,X_dev,Y_dev,0.0)

# As a rough sanity check, our loss should be something close to -log(0.1).
print('loss: '+str(loss))
print('sanity check: '+str(-np.log(0.1)))
# the reason is that: there are 10 classes, 
#and the averge of Loss if approximately equal to  -np.log(0.1)

# As we did for the SVM, use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient.
loss,grad=softmax.softmax_loss_naive(W,X_dev,Y_dev,0.0)
f=lambda w:softmax.softmax_loss_naive(w,X_dev,Y_dev,0.0)[0]
Tools.grad_check_sparse(f,W,grad,10)

# similar to SVM case, do another gradient check with regularization
loss,grad=softmax.softmax_loss_naive(W,X_dev,Y_dev,5e1)
f=lambda w:softmax.softmax_loss_naive(w,X_dev,Y_dev,5e1)[0]
Tools.grad_check_sparse(f,W,grad,10)


# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.
tic = time.time()
loss_naive, grad_naive = softmax.softmax_loss_naive(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

f
tic = time.time()
loss_vectorized, grad_vectorized = softmax.softmax_loss_vectorized(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# As we did for the SVM, we use the Frobenius norm to compare the two versions
# of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
print('Gradient difference: %f' % grad_difference)

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.
results={}
best_val=-1
best_softmax=None
learning_rates=[1e-7,5e-7]
regularization_strengths=[2.5e4,5e4]


for lr in learning_rates:
    for rs in regularization_strengths:
        softmax=liner_classifiers_total.Softmax()
        
        loss_history=softmax.train(X_train,Y_train,lr,rs,num_iters=1500,batch_size=100,verbose=True)
        Y_train_pred=softmax.predict(X_train)
        train_accuracy=np.mean(Y_train==Y_train_pred)
        
        Y_val_pred=softmax.predict(X_val)
        val_accuracy=np.mean(Y_val==Y_val_pred)
        
        if val_accuracy>best_val:
            best_val=val_accuracy
            best_softmax=softmax
        results[(lr,rs)]=train_accuracy,val_accuracy

# print out result
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)


# evaluate on test set
# Evaluate the best softmax on test set
Y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(Y_test == Y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))



# Visualize the learned weights for each class
W=best_softmax.W[:-1,:]  # strip out the bias
W=W.reshape(32,32,3,10)

W_min,W_max=np.min(W),np.max(W)

classes=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2,5,i+1)
     # Rescale the weights to be between 0 and 255
    wimg=255*(W[:,:,:,i].squeeze()-W_min)/(W_max-W_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])


























