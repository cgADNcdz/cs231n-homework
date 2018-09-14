#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 20:01:38 2018

@author: cdz
"""
import numpy as np
import matplotlib.pyplot as plt
import Tools
import SVM
import time
from liner_classifiers_total import LinearSVM


###********************************************************************************************
#******************************************  test SVM  ******************************************

# set default size of plots
plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'


#########################  Data loading and processing

#load data (cubic type)
X_train,Y_train,X_test,Y_test=Tools.load_CIFAR10('cifar-10-batches-py')
print('X_train shape: '+str(X_train.shape))
print('Y_train shape: '+str(Y_train.shape))
print('X_test shape: '+str(X_test.shape))
print('Y_test shape: '+str(Y_test.shape))

'''
# visualize some examples from the dataseet
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes=len(classes)
sample_per_class=7
for y,cls in enumerate(classes):  # y is number, cls is name
    idxs=np.flatnonzero(Y_train==y)  # return indices where parameters is not zero
    idxs=np.random.choice(idxs,sample_per_class,replace=False)
    for i,idx in enumerate(idxs):
        plt_idx=i*num_classes+y+1
        plt.subplot(sample_per_class,num_classes,plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cls)
plt.show()
'''

# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training=49000
num_validation=1000
num_test=1000
num_dev=500

mask=range(num_training,num_training+num_validation)
X_val=X_train[mask]
Y_val=Y_train[mask]

mask=range(num_training)
X_train=X_train[mask]
Y_train=Y_train[mask]

mask=np.random.choice(num_training,num_dev,replace=False)  #choose randomly
X_dev=X_train[mask]
Y_dev=Y_train[mask]

mask=range(num_test)
X_test=X_test[mask]
Y_test=Y_test[mask]
print('\n\n after split')
print('X_train shape: '+str(X_train.shape))
print('Y_train shape: '+str(Y_train.shape))
print('X_val shape: '+str(X_val.shape))
print('Y_val shape: '+str(Y_val.shape))
print('X_test shape: '+str(X_test.shape))
print('Y_test shape: '+str(Y_test.shape))


# Preprocessing: reshape the image data into rows
X_train=np.reshape(X_train,(X_train.shape[0],-1))
X_val=np.reshape(X_val,(X_val.shape[0],-1))
X_test=np.reshape(X_test,(X_test.shape[0],-1))
X_dev=np.reshape(X_dev,(X_dev.shape[0],-1))
print('\n\n after reshape')
print('X_train shape: '+str(X_train.shape))
print('X_val shape: '+str(X_val.shape))
print('X_test shape: '+str(X_test.shape))
print('X_dev shape: '+str(X_dev.shape))


# Preprocessing: subtract the mean image
mean_image=np.mean(X_train,axis=0)   # first: compute the image mean based on the training data
'''
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8'))  # visualize the mean image
plt.show()
'''
X_train-=mean_image                  # second: subtract the mean image from train and test data
X_val-=mean_image
X_test-=mean_image
X_dev-=mean_image

X_train=np.hstack([X_train,np.ones((X_train.shape[0],1))])    # third: append the bias dimension of ones
X_val=np.hstack([X_val,np.ones((X_val.shape[0],1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
print('\n\n after apend bias')
print('X_train shape: '+str(X_train.shape))
print('X_val shape: '+str(X_val.shape))
print('X_test shape: '+str(X_test.shape))
print('X_dev shape: '+str(X_dev.shape))



'''
############################     SVM classifier
classifier=SVM.SVM()

# generate a random SVM weight matrix of small numbers
W=np.random.randn(3073,10)*0.0001

loss,grad=classifier.svm_loss_naive(W,X_dev,Y_dev,0.00001)
print('\n\n loss: '+str(loss))
print('dW:')
print(grad)

# Compute the loss and its gradient at W.
loss,grad=classifier.svm_loss_naive(W,X_dev,Y_dev,0.)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
f=lambda W:classifier.svm_loss_naive(W,X_dev,Y_dev,0.0)[0]  # lambda expression  f=lambda input:output
Tools.grad_check_sparse(f,W,grad)


# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
loss, grad =classifier.svm_loss_naive(W, X_dev, Y_dev, 1e2)
f = lambda w: classifier.svm_loss_naive(w, X_dev, Y_dev, 1e2)[0]
grad_numerical = Tools.grad_check_sparse(f, W, grad)




# The naive implementation and the vectorized implementation should match, but
# the vectorized version should still be much faster.
tic = time.time()
_, grad_naive = classifier.svm_loss_naive(W, X_dev, Y_dev, 0.00001)
toc = time.time()
print ('Naive loss and gradient: computed in %fs' % (toc - tic))

tic = time.time()
_, grad_vectorized = classifier.svm_loss_vectorized(W, X_dev, Y_dev, 0.00001)
toc = time.time()
print('Vectorized loss and gradient: computed in %fs' % (toc - tic))

# The loss is a single number, so it is easy to compare the values computed
# by the two implementations. The gradient on the other hand is a matrix, so
# we use the Frobenius norm to compare them.
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print ('difference: %f' % difference)
'''


###############################   SGD 
'''
svm=LinearSVM()
tic=time.time()
loss_history=svm.train(X_train,Y_train,learning_rate=1e-7,regularization=5e4,
                       num_iters=1500,verbose=True)
toc=time.time()
print('take time: '+str(toc-tic))


# A useful debugging strategy is to plot the loss as a function of
# iteration number
plt.plot(loss_history)
plt.xlabel('Iteration number')
plt.ylabel('loss value')
plt.show()


# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
Y_train_pred = svm.predict(X_train)
print ('training accuracy: %f' % (np.mean(Y_train == Y_train_pred), ))
Y_val_pred = svm.predict(X_val)
print( 'validation accuracy: %f' % (np.mean(Y_val == Y_val_pred), ))
'''

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
learning_rates=[1.4e-7,1.5e-7,1.6e-7]
regularization_strengths=[(1+i*0.1)*1e4 for i in range(-3,3)] + [(2+0.1*i)*1e4 for i in range(-3,3)]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results={}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

for rs in regularization_strengths:
    for lr in learning_rates:
        svm=LinearSVM()
        loss_history=svm.train(X_train,Y_train,lr,rs,num_iters=3000)
        
        Y_train_pred=svm.predict(X_train)
        train_accuracy=np.mean(Y_train==Y_train_pred)
        
        Y_val_pred=svm.predict(X_val)
        val_accuracy=np.mean(Y_val==Y_val_pred)
        
        if val_accuracy>best_val:
            best_val=val_accuracy
            best_svm=svm
        results[(lr,rs)]=train_accuracy,val_accuracy



# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print ('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print ('best validation accuracy achieved during cross-validation: %f' % best_val)



# Visualize the cross-validation results
import math
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()


# Evaluate the best svm on test set
Y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(Y_test == Y_test_pred)
print ('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)


# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = best_svm.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
  plt.subplot(2, 5, i + 1)
    
  # Rescale the weights to be between 0 and 255
  wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
  plt.imshow(wimg.astype('uint8'))
  plt.axis('off')
  plt.title(classes[i])


















