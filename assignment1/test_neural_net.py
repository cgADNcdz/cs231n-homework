#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 19:50:52 2018

@author: cdz
"""

import numpy as np
import matplotlib.pyplot as plt
import Two_layer_Neural_Net
import Tools

plt.rcParams['figure.figsize']=(10.0,8.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

def rel_error(x,y):
    '''
    return relative error
    '''   # i think no need to use np.max
    return np.max(np.abs(x-y)/(np.maximum(1e-8,np.abs(x)+np.abs(y))))
    
# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.
    
input_size=4
hidden_size=10
num_classes=3
num_inputs=5

'''
def init_toy_model():
    np.random.seed(0)
    return Two_layer_Neural_Net.Two_layer_net(input_size,hidden_size,num_classes,std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X=10*np.random.randn(num_inputs,input_size)
    y=np.array([0,1,2,2,1])
    return X,y

net=init_toy_model()
X,y=init_toy_data()

# forward pass, compute the scores
scores=net.loss(X)
print('scores')
print(scores)
correct_scores = np.asarray([    # given by teachers
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print('correct scores')
print(correct_scores)
print('difference:')
print(np.sum(np.abs(scores,correct_scores)))


# forward pass, compute loss
loss,_=net.loss(X,y,regularization=0.1)
correct_loss = 1.30378789133
print(loss)
print('difference')
print(np.sum(np.abs(loss-correct_loss)))

# backward pass
# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss,grads=net.loss(X,y,regularization=0.1)
# these should all be less than 1e-8 or so
for param_name in grads:  #  check the gradient of W1 W2, b1 b2
    f=lambda W:net.loss(X,y,regularization=0.1)[0]
    param_grad_num=Tools.eval_numerical_gradient(f,net.params[param_name],verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))                       


# train the net 
net=init_toy_model()
stats=net.train(X,y,X,y,learning_rate=1e-1,regularization=1e-5,num_iterations=100,verbose=False)
print('Final training loss: ',stats['loss_history'][-1])
#plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('training loss history')
plt.show()

'''

########################################   test network
  # Load the raw CIFAR-10 data
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test =Tools.load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', y_val.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)



input_size = 32 * 32 * 3
num_classes = 10
'''
hidden_size = 50

net = Two_layer_Neural_Net.Two_layer_net(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iterations=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.5, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print ('Validation accuracy: ', val_acc)

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()
'''

########Tune hyperparameters
best_net = None # store the best model into this

hidden_size = [75, 100, 125]

results = {}
best_val_acc = 0
best_net = None

learning_rates = np.array([0.7, 0.8, 0.9, 1, 1.1])*1e-3
regularization_strengths = [0.75, 1, 1.25]

print ('running')
for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            print ('.')
            net =Two_layer_Neural_Net.Two_layer_net(input_size, hs, num_classes)
            # Train the network
            stats = net.train(X_train, y_train, X_val, y_val,
            num_iterations=1500, batch_size=200,
            learning_rate=lr, learning_rate_decay=0.95,
            reg= reg, verbose=False)
            val_acc = (net.predict(X_val) == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net         
            results[(hs,lr,reg)] = val_acc 
print ("finshed")
# Print out results.
for hs,lr, reg in sorted(results):
    val_acc = results[(hs, lr, reg)]
    print ('hs %d lr %e reg %e val accuracy: %f' % (hs, lr, reg,  val_acc))
    
print ('best validation accuracy achieved during cross-validation: %f' % best_val_acc)


test_acc = (best_net.predict(X_test) == y_test).mean()
print ('Test accuracy: '+str(test_acc))











