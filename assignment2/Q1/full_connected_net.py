#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 22:23:56 2018

@author: cdz
"""
import time 
import numpy as np
import matplotlib.pyplot as plt
import layers
import Tools
import layer_utilities
import classifier_fc_net

plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['image.interpolation']='neartest'
plt.rcParams['image.cmap']='gray'

'''
# test affine_forward function
num_inputs=2 # number of samples
input_shape=(4,5,6)  #dimension of each sample 4*5*6
output_dim=3

input_size=num_inputs*np.prod(input_shape) # 2*120
weight_size=output_dim*np.prod(input_shape) # 3*120

x=np.linspace(-0.1,0.5,num=input_size).reshape(num_inputs,*input_shape)  # * menas tuple
w=np.linspace(-0.2,0.3,num=weight_size).reshape(np.prod(input_shape),output_dim)                                                          
b=np.linspace(-0.3,0.1,num=output_dim)   # start, stop, number

out,_=layers.affine_forward(x,w,b)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])
# we can konw that: input units=120, output units=3, x.shape=2 by 120, w.shape=120 by 3, b.shape=(3,)
print('test affine_forward function:')
print('difference: '+str(Tools.rel_error(out,correct_out)))
'''

'''
#test affine_backward function
np.random.seed(231)
x=np.random.randn(10,2,3)
w=np.random.randn(6,5)
b=np.random.randn(5)
dout=np.random.randn(10,5)

dx_num=Tools.eval_numerical_gradient_array(lambda x:layers.affine_forward(x,w,b)[0],x,dout)
dw_num=Tools.eval_numerical_gradient_array(lambda w:layers.affine_forward(x,w,b)[0],w,dout)
db_num=Tools.eval_numerical_gradient_array(lambda b:layers.affine_forward(x,w,b)[0],b,dout)

_,cache=layers.affine_forward(x,w,b)
dx,dw,db=layers.affine_backward(dout,cache)

print('difenence:')
print('dx error: ',Tools.rel_error(dx_num,dx))
print('dw error: ',Tools.rel_error(dw_num,dw))
print('db error: ',Tools.rel_error(db_num,db))
'''
'''
# test relu_forward function
x=np.linspace(-0.5,0.5,num=12).reshape(3,4)
out,_=layers.relu_forward(x)

correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                        [ 0.,          0.,          0.04545455,  0.13636364,],
                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

print('testing relu_forward')
print('difference: ', Tools.rel_error(out, correct_out))
'''

'''
# test relu_backward
np.random.seed(231)
x=np.random.randn(10,10)
dout=np.random.randn(*x.shape)

dx_num=Tools.eval_numerical_gradient_array(lambda x:layers.relu_forward(x)[0],x,dout)
_,cache=layers.relu_forward(x)
dx=layers.relu_backward(dout,cache)
 
# The error should be on the order of e-12
print('Testing relu_backward function:')
print('dx error: ', Tools.rel_error(dx_num, dx))
'''

'''
np.random.seed(231)
x = np.random.randn(2, 3, 4)
w = np.random.randn(12, 10)
b = np.random.randn(10)
dout = np.random.randn(2, 10)

out, cache = layer_utilities.affine_relu_forward(x, w, b)
dx, dw, db = layer_utilities.affine_relu_backward(dout, cache)

dx_num = Tools.eval_numerical_gradient_array(lambda x: layer_utilities.affine_relu_forward(x, w, b)[0], x, dout)
dw_num = Tools.eval_numerical_gradient_array(lambda w: layer_utilities.affine_relu_forward(x, w, b)[0], w, dout)
db_num = Tools.eval_numerical_gradient_array(lambda b: layer_utilities.affine_relu_forward(x, w, b)[0], b, dout)

# Relative error should be around e-10 or less
print('Testing affine_relu_forward and affine_relu_backward:')
print('dx error: ', Tools.rel_error(dx_num, dx))
print('dw error: ', Tools.rel_error(dw_num, dw))
print('db error: ', Tools.rel_error(db_num, db))
'''

'''
#test softmax and svm loss layer
np.random.seed(231)
num_classes, num_inputs = 10, 50
x = 0.001 * np.random.randn(num_inputs, num_classes)
y = np.random.randint(num_classes, size=num_inputs)

dx_num =Tools.eval_numerical_gradient(lambda x: layers.svm_loss(x, y)[0], x, verbose=False)
loss, dx =layers.svm_loss(x, y)

# Test svm_loss function. Loss should be around 9 and dx error should be around the order of e-9
print('Testing svm_loss:')
print('loss: ', loss)
print('dx error: ',Tools.rel_error(dx_num, dx))

dx_num =Tools.eval_numerical_gradient(lambda x: layers.softmax_loss(x, y)[0], x, verbose=False)
loss, dx =layers.softmax_loss(x, y)

# Test softmax_loss function. Loss should be close to 2.3 and dx error should be around e-8
print('\nTesting softmax_loss:')
print('loss: ', loss)
print('dx error: ', Tools.rel_error(dx_num, dx))
'''

'''
# test TwoLayerNet
np.random.seed(231)
N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)

std = 1e-3
model = classifier_fc_net.TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

print('Testing initialization ... ')
W1_std = abs(model.params['W1'].std() - std) # std标准差
b1 = model.params['b1']
W2_std = abs(model.params['W2'].std() - std)
b2 = model.params['b2']
assert W1_std < std / 10, 'First layer weights do not seem right'
assert np.all(b1 == 0), 'First layer biases do not seem right'
assert W2_std < std / 10, 'Second layer weights do not seem right'
assert np.all(b2 == 0), 'Second layer biases do not seem right'

print('Testing test-time forward pass ... ')
model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
scores = model.loss(X)
correct_scores = np.asarray(
  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
scores_diff = np.abs(scores - correct_scores).sum()
assert scores_diff < 1e-6, 'Problem with test-time forward pass'

print('Testing training loss (no regularization)')
y = np.asarray([0, 5, 1])
loss, grads = model.loss(X, y)
correct_loss = 3.4702243556
assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

model.reg = 1.0
loss, grads = model.loss(X, y)
correct_loss = 26.5948426952
assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

# Errors should be around e-7 or less
for reg in [0.0, 0.7]:
  print('Running numeric gradient check with reg = ', reg)
  model.reg = reg
  loss, grads = model.loss(X, y)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = Tools.eval_numerical_gradient(f, model.params[name], verbose=False)
    print('%s relative error: %.2e' % (name, Tools.rel_error(grad_num, grads[name])))
'''    








































