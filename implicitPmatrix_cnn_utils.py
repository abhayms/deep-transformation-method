# not sure if this is official, but seems proper, use only some methods from this

# 30_4 currently the random batches are disabled. batches are strictly interleaved

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

# to use this data will be have to be in this format
def load_dataset():


    trafile = 'DTM_train.h5'
    testfile= 'DTM_test.h5'



    train_dataset = h5py.File(trafile, "r") # data.h5- actually only train data acc to our code
    
        
    train_set_x_orig = np.array(train_dataset["traintensor"][:]) # maybe outside np.array not required
    train_set_y_orig = np.array(train_dataset["trainlabel"][:]) # your train set labels

    Pmatrix= np.array(train_dataset["Pmatrix"][:])
    #TODO  CHECK if the data from mat is being reproduced here
    test_dataset = h5py.File(testfile, "r")
    test_set_x_orig = np.array(test_dataset["testtensor"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["testlabel"][:]) # your test set labels

    #classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    classes= np.array([1,0]) # check what format is expected 6_4
    
    #already stored in this shape    
    #train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) # CHECK HERE 6/4
    #test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes,trafile # CHANGE 12_4- also returning file name


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0): # note default is 64
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y) - anyway permuation hotay so we don't need to do it at start
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:] # just permute directly on batch index!!!
    shuffled_Y = Y[permutation,:] # VERIFIED, both the labels have been shuffled properly according to X.

############### CHANGE 29_4 removed random permutation to check  interleaving  
    #shuffled_X = X # just permute directly on batch index!!!
    #shuffled_Y = Y# VERIFIED, both the labels have been shuffled properly according to X.

############################


    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = np.uint32(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:] # sopay, just slide along
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T # select appropriate rows from ident matrix dep on the raw label vector
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

# not used in code
def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction
