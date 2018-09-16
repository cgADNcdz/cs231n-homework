#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 16:01:00 2018

@author: cdz
"""
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import time
from Tools import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

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

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)



# clear old variables
tf.reset_default_graph()

# setup input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

def simple_model(X,y):
    # define our weights (e.g. init_two_layer_convnet)
    
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 10])
    b1 = tf.get_variable("b1", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1,[-1,5408])
    y_out = tf.matmul(h1_flat,W1) + b1
    return y_out

y_out = simple_model(X,y)

# define our loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

'''
with tf.Session() as sess:
    with tf.device("/gpu:0"): #"/cpu:0" or "/gpu:0" 
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
        print('Validation')
        run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
  '''



######  trining a specific model
# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in 
X=tf.placeholder(tf.float32,[None,32,32,3])
y=tf.placeholder(tf.int64,[None])
is_training=tf.placeholder(tf.bool)

#define model
def complex_model(X,y,id_training):
    # define our weights (e.g. init_two_layer_convnet)
    
    # setup variables
    #conv layer
    Wconv1=tf.get_variable("Wconv1",shape=[7,7,3,32])
    bconv1=tf.get_variable("bconv1",shape=[32])
    
    # affine1 layer 
    W1=tf.get_variable("W1",shape=[5408,1024])  # 5408=13*13*32 
    b1=tf.get_variable("b1",shape=[1024])
    
    # affine2 layer
    W2=tf.get_variable("W2",shape=[1024,10])
    b2=tf.get_variable("b2",shape=[10])
    
    # define our graph (e.g. two_layer_convnet)
    a1=tf.nn.conv2d(X,Wconv1,strides=[1,1,1,1],padding='VALID')+bconv1
    h1=tf.nn.relu(a1)
    
    #spatial batch normalization
    num_filters=32
    # compute on which dimensions
    axes=[0,1,2]
    mean,var=tf.nn.moments(h1,axes=axes)
    offset=tf.Variable(tf.zeros([num_filters]))
    scale=tf.Variable(tf.ones([num_filters]))
    eps=0.0001
    bn1=tf.nn.batch_normalization(h1,mean,var,offset=offset,scale=scale,variance_epsilon=eps)
   
    pool1=tf.nn.max_pool(bn1,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
    p1_flat = tf.reshape(pool1,[-1,5408])
    y1_out = tf.matmul(p1_flat,W1) + b1
    
    
    h2 = tf.nn.relu(y1_out)
    h2_flat = tf.reshape(h2,[-1,1024])
    y2_out=tf.matmul(h2_flat,W2)+b2
    
    return y2_out



y_out=complex_model(X,y,is_training)
# Now we're going to feed a random batch into the model 
# and make sure the output is the right size
x=np.random.randn(64,32,32,3)
'''
with tf.Session() as sess:
    with tf.device('/cpu:0'):
        tf.global_variables_initializer().run()
        
        ans=sess.run(y_out,feed_dict={X:x,is_training:True})
        print(ans.shape)
        print(np.array_equal(ans.shape, np.array([64, 10])))
'''
# train the model
# Inputs
#     y_out: is what your model computes
#     y: is your TensorFlow variable with label information
# Outputs
#    mean_loss: a TensorFlow variable (scalar) with numerical loss
#    optimizer: a TensorFlow optimizer
# This should be ~3 lines of code!
mean_loss = None
optimizer = None
total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
optimizer=tf.train.RMSPropOptimizer(1e-3)


# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)
'''
sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step)

print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
'''



###################################################################
##############  train a great model on cifar-10
# Feel free to play with this cell

def conv33_relu_batch(inputs,num_filter):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.2)
    conv1 = tf.layers.conv2d(
      inputs=inputs,
      filters=num_filter,
      strides=[1,1],
      kernel_size=[3, 3],
      padding="same",
      kernel_regularizer=regularizer)
    
    relu1=tf.nn.relu6(conv1)
    batch_norm1=tf.layers.batch_normalization(relu1,axis=1)
    dropout1 = tf.layers.dropout(inputs=batch_norm1, rate=0.5)
    
    conv2 = tf.layers.conv2d(
      inputs=dropout1,
    #when filters=2*num_filters,it achieve results follows just after 20 epochs
        #val:(0.82898137760162349, 0.85199999999999998)
        #test:(0.93189846858978276, 0.82850000000000001)
        
      filters=num_filter*2,
      strides=[1,1],
      kernel_size=[3, 3],
      padding="same",
      kernel_regularizer=regularizer)
    
    relu2=tf.nn.relu6(conv2)
    batch_norm2=tf.layers.batch_normalization(relu2,axis=1)
    
    pool= tf.layers.max_pooling2d(inputs=batch_norm2, pool_size=[2, 2], strides=2)
    dropout2 = tf.layers.dropout(inputs=pool, rate=0.5)
    return dropout2

def my_model(X,y,is_training):
    #structure
    #[conv-relu-conv-relu] -> global average pooling -> [softmax]
    num_classes=10
    # define our graph use layer API
    nn1=conv33_relu_batch(X,64)
    nn2=conv33_relu_batch(nn1,128)
    nn3=conv33_relu_batch(nn2,256)
    #global average pooling
    pool_size=(nn3.shape[1],nn3.shape[2])
    pool_ave = tf.layers.average_pooling2d(
      inputs=nn3,
      pool_size=pool_size,
      strides=[1,1],
      padding='valid',
      data_format='channels_last')
    pool_ave_flat_size=pool_ave.shape[1]*pool_ave.shape[2]*pool_ave.shape[3]
    pool_ave_flat = tf.reshape(pool_ave, [-1,pool_ave_flat_size])
    dropout = tf.layers.dropout(inputs=pool_ave_flat, rate=0.5, training=is_training)
    
    y_out= tf.layers.dense(inputs=dropout, units=num_classes)
    return y_out

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X,y,is_training)
total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

#learning rate decay
global_step = tf.Variable(0, trainable=False)
initial_learning_rate=1e-2
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=50,decay_rate=0.9)

optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

time1=time.time()
# Feel free to play with this cell
# This default code creates a session
# and trains your model for 10 epochs
# then prints the validation set accuracy
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,101):
    print('Training epoch%d '%(i+1))
    run_model(sess,y_out,mean_loss,X_train,y_train,1,64,500,train_step,True)
    print('Validation')
    run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
    print()
    print()
    
    
    
#Test your model here, and make sure 
# the output of this cell is the accuracy
# of your best model on the training and val sets
# We're looking for >= 70% accuracy on Validation
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,1,64)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)

print('Test')
run_model(sess,y_out,mean_loss,X_test,y_test,1,64)

time2=time.time()
print("total time:",time2-time1)















































