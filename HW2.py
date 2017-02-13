"""
Author:Benjamin Fields
Program: HW2.py
Date:2/12/2017
Description:Implementation of a one hidden layer 
fully connected neural network.
This network will be used to classify two classes 
of the widely recognized cifar-10 data set. This example 
uses a cifar-2 dataset with airplanes and ships
"""


from __future__ import division
from __future__ import print_function

import sys
import matplotlib.pyplot as plt

import _pickle as cPickle
import numpy as np

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
	    # DEFINE __init function
        self.W = W
        self.b = b
        

    def forward(self, x):
        return np.sum(x*self.W)+self.b


    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        pass
	# DEFINE backward function
    # ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def forward(self, x):
        return np.maximum(x,0)


    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
    # DEFINE backward function
    # ADD other operations in ReLU if needed
        pass


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x,y):
        return -y*np.log(x)-(1-y)*np.log(1-x)
        
        

    def backward(self,grad_output,learning_rate=0.0,momentum=0.0,l2_penalty=0.0):

		# DEFINE backward function
        # ADD other operations and data entries in SigmoidCrossEntropy if needed
        pass


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
        # INSERT CODE for initializing the network
        self.W1 = np.random.rand(hidden_units,input_dims)
        self.W2 = np.random.rand(2,hidden_units)
        self.b = np.random.rand()
        self.c = np.random.rand()
        self.layerOne = LinearTransform(self.W1,self.b)
        self.layerTwo = LinearTransform(self.W2,self.c)
        


    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty,
    ):
	    # INSERT CODE for training the network
        pass
    def softmax(self,x):
        return 1/(1+np.exp(-x))

    def forwardCalc(self,x):
        first = ReLU()
        firstActive = first.forward(self.layerOne.forward(x))
        #print("First Activation {}".format(firstActive))
        secondActive = first.forward(self.layerTwo.forward(firstActive))
        #print("Second Activation {}".format(secondActive))

        return self.softmax(secondActive)
    
    
    
    def evaluate(self, x, y):
        loss = x-y
        return loss.astype(float)


if __name__ == '__main__':

    #data = cPickle.load(open('cifar_2class_py2.p', 'rb'))
    data = cPickle.load(open('cifar_2class_py2.p', 'rb'), encoding='iso-8859-1')

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
    num_examples, input_dims = train_x.shape
    print('Number of examples: {}'.format(num_examples))
    print('input dims: {}'.format(input_dims))
    #print(train_x[1])
    #print(train_y)

	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    num_batches = 1000
    mlp = MLP(3072, 5)
    
    first = mlp.forwardCalc(train_x[0])
    
    print('Predicted: {} actual: {}'.format(first,train_y[0]))
    print("Loss is {}".format(mlp.evaluate(first,train_y[0])))
   
    index = 0
    for epoch in range(num_epochs):

	# INSERT YOUR CODE FOR EACH EPOCH HERE
        train_loss = 0.0

        for b in range(num_batches):
            total_loss = 0.0
            
			    # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            total_loss += float(mlp.evaluate(mlp.forwardCalc(train_x[index]),train_y[index]))
            train_loss += total_loss
                
			# MAKE SURE TO UPDATE total_loss
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss/float(b+1),
                ),
                end='',
            )
            sys.stdout.flush()
            index+=1
            
		# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
		# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        
        train_accuracy = 0.0
        test_loss = 0.0
        test_accuracy = 0.0
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))