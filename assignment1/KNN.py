#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:11:03 2018

@author: cdz
"""

import numpy as np
import time
import Tools

'''
1.尽可能向量化, reference: http://www.hellocg.xin:81/wordpress/?p=127
2.矩阵太大，Python通常只是单核，应该调整代码，使用多进程
3.KNN准确率并不高(may be trianing data is too little)
'''

class KNN(object):
    def _init(self):
        self.X_train   #member variables
        self.y_train
    
    def train(self,X,y):
        self.X_train=X
        self.y_train=y
    
    def predict(self,X_test,K=1,num_loops=0):
        '''
        X:test samples
        K:number of nearst neighbors
        num_loops:determine how to compute distances
        
        return labels
        '''
        if num_loops==0:
            dists=self.compute_distances_no_loop(X_test)
        elif num_loops==1:
            dists=self.compute_distances_one_loop(X_test)
        elif num_loops==2:
            dists=self.compute_distances_two_loops(X_test)
        else:
            raise ValueError("INvalid value for num_loops"+str(num_loops))
        return self.predict_labels(dists,K)
    
    
    # L2 distance using two loops to compute
    def compute_distances_two_loops(self,X_test):
        '''
        X_test:an array of shape(num_test,D)
        return an array of shape(mun_test,num_train),  dists[i, j] is the Euclidean distance 
        between the ith test point and the jth training point.
        '''
        num_test=X_test.shape[0]   #rows
        num_train=self.X_train.shape[0]
        dists=np.zeros((num_test,num_train))
        for i in range(num_test):
            for j in range(num_train):
                vec_temp=X_test[i,:]-self.X_train[j,:]  #subtract of two vector
                #vec_temp_t.shape=(X_test.shape[1],1)  #transpose,become col vector(no need, array can dot directly)
                #dists[i,j]=np.sqrt(vec_temp.dot(vec_temp))
                dists[i,j]=np.linalg.norm(vec_temp,ord=2)
        return dists
    
    # L2 distance using oen loop to compute
    def compute_distances_one_loop(self,X_test):
        '''
        X_test: an array of shape(num_test,D)
        return an array of shape(mun_test,num_train),  dists[i, j] is the Manhattan distance
        between the ith test point and the jth training point.
        '''
        num_test=X_test.shape[0]   #rows
        num_train=self.X_train.shape[0]
        dists=np.zeros((num_test,num_train))
        for i in range(num_test):
            mat_temp=self.X_train-X_test[i,:] #10000x3072
            dists[i,:]=np.linalg.norm(mat_temp,axis=1,ord=2) # L2 norm
        return dists
    
    # L2 distance using no loop
    def compute_distances_no_loop(self,X_test):
        '''
        X_test: an array of shape(num_test,D)
        return an array of shape(mun_test,num_train),  dists[i, j] is the ...... distance
        between the ith test point and the jth training point.
        '''
        num_test=X_test.shape[0]   #rows
        num_train=self.X_train.shape[0]
        dists=np.zeros((num_test,num_train))
        ABT=X_test.dot(self.X_train.transpose())  # reference: http://www.hellocg.xin:81/wordpress/?p=127
        
        Aml=X_test**2
        Aml=np.matrix(np.sum(Aml,axis=1))
        Aml=np.tile(Aml.transpose(),(1,X_train.shape[0]))  # attention transpose
        Aml=np.array(Aml)
        
        Bnl=self.X_train**2
        Bnl=np.matrix(np.sum(Bnl,axis=1))  # must use matrix format 
        Bnl=np.tile(Bnl,(X_test.shape[0],1))
        Bnl=np.array(Bnl)  #come back to array
       
        dists=Aml+Bnl-2*ABT    # (a-b)^2=a^2+b^2-2ab
        return dists
    
    
    def predict_labels(self,dists,K=1):
        '''
        dists: the dists[i,j] is the distance between the ith test_dataset and the jth train dataset
        K:the number of the nearest points
        return y, an array of shape(num_test,1),y[i] is the label of ith testdataset
        '''
        num_test=dists.shape[0]
        y_pred=np.zeros(num_test)
        for i in range(num_test):
            closest_y=[]
            index_k=np.argsort(dists[i,:])[0:K]  # get the inxex of nearest K points
            closest_y=np.array(self.y_train)[index_k] 
            counts=np.bincount(closest_y)
            y_pred[i]=np.argmax(counts)

        return y_pred
            
            
#******************* run ******************************************
        
tools=Tools.Tools()
KNN=KNN()

batch1=tools.unpickle('cifar-10-batches-py/data_batch_1') 
#batch2=tools.unpickle('cifar-10-batches-py/data_batch_2')
#batch3=tools.unpickle('cifar-10-batches-py/data_batch_3')
#batch4=tools.unpickle('cifar-10-batches-py/data_batch_4')
#batch5=tools.unpickle('cifar-10-batches-py/data_batch_5')
#X_train=np.concatenate((batch1[b'data'],batch2[b'data'],batch3[b'data'],batch4[b'data'],batch5[b'data']),axis=0)                                   
#y_train=np.concatenate((batch1[b'labels'],batch2[b'labels'],batch4[b'labels'],batch4[b'labels'],batch5[b'labels']))
X_train=batch1[b'data']
y_train=batch1[b'labels']

#print(X_train.shape)
#print(y_train.shape)

test_batch=tools.unpickle('cifar-10-batches-py/test_batch')
X_test=test_batch[b'data'][0:2000,:]
y_test=test_batch[b'labels'][0:2000]


KNN.train(X_train,y_train)
t1=time.time()
labels=KNN.predict(X_test,K=20,num_loops=0)
t2=time.time()
print('time total: '+str(t2-t1))

#caculate accuracy 
diff=np.array(y_test)-labels
same_num=np.sum(diff==0)
accuracy=same_num/len(y_test)
print("accuracy: "+str(accuracy))



#test functions
        
''' get data
tool=Tools() 
meta=tool.get_meta('cifar-10-batches-py/batches.meta')
print('mum_vis:'+str(meta[b'num_vis']))
print('num_cases_per_batch'+str(meta[b'num_cases_per_batch']))
n=len(meta[b'label_names'])
for i in range(n):
    print('label'+str(i)+':'+str(meta[b'label_names'][i])) 
 '''
    
 

'''  numpy array caculation
x=np.array([(1,2,3,4),(6,7,8,9)])
print(x[1,:]-x[0,:])  #can subtract directly
m=x[0,:]
n=x[1,:]
#m=np.transpose(m)  //can not transpose one dimension
m.shape=(4,1) 
y=n.dot(m)  #dot(), ranther than * 
print(y)

xx=np.array([-1,2,-3,-5])
print(np.abs(xx))  #can get absolute value of a vector directly
print(sum(xx))   #summary of an array
    
b=np.zeros(3) #one dimension array
print(b)   

c=[]  # test type of [] 
print(type(c)) 

d=np.array([2,3,5,6,8,1,4,9])
e=np.argsort(d) #argsort: return the index of element in ascending order
print(e)
print(e[0:6])  #left close and right open
print(d[e]) #get the element of an array by a list of index
 
f=[2,3,4,5,6,3,3,3,3,3,3,3,3,4]
counts = np.bincount(f)  # get the mode(众数)
print(counts)
g=np.argmax(counts)
print(g)

h=np.array([(2,3,5,1),(5,7,9,8)])
i=np.array([1,3,2,4])
#print(h.transpose())
print((h-i).transpose())
 
j=np.linalg.norm(h,axis=1,ord=2) #norm of a array(vector),axis=1 means conpute row vector
print("j: "+str(j))
 
h=h**2                   
print(h)
h=np.matrix(np.sum(h,axis=1))
print(h)
h=np.tile(h.transpose(),(1,3))
print(h)
print(type(h))
h=np.array(h)
print(h)
print(type(h))
print(h.shape)
'''
 














