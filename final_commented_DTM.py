# coding: utf-8

#################
 #The tensorflow code which implements proposed Deep Transformation Method (DTM. It contains the implementation of the transformation layer
 #which produces cross-correlation matrices for the input time series data. This is followed by convolution- ReLu- Max pool modules and
 #fully connected layer for use in time series classification problems.

#We generate results on the ADHD-200 fMRI data for classifying ADHD and TDC(typically developing children)subjects.

#The proposed method gives an average classification accuracy of 70.30 %    

##############


import os

# set up GPU
os.environ["CUDA_VISIBLE_DEVICES"]="7" # 5

########## import useful modules ###############
import math
import numpy as np
import h5py
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import seaborn as sns

#import scipy
from PIL import Image
#from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
#from implicit_cnn_utils import *
from implicit_cnn_utils import *
sns.set_context("paper")
sns.set_style("whitegrid", {"grid.color": ".92","ytick.major.size":"5","xtick.major.size":"5","legend.frameon":"False","axes.edgecolor":"0.1","axes.linewidth": "2","axes.labelcolor": "0.1","text.color": "0.1"})      


# fix the random seed
np.random.seed(1)

print('you are looking at correct file...')

# Loading the stored input data (X_train_orig is training time series matrix of 90 x T_n and Y_train_orig is a 90x 1 vector indicating
# 0: TDC and 1: ADHD, X_test and Y_test are the test matrices  
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes, dataname = load_dataset() 

square_dim=90 # the number of brain regions in the input data

import PIL.Image as pil


# To get started, checking the shapes of the input data. 

X_train = X_train_orig
X_test = X_test_orig

# convert Y to one hot encoding 
Y_train = convert_to_one_hot(np.uint8(Y_train_orig), 2).T  #np.uint8
Y_test = convert_to_one_hot(np.uint8(Y_test_orig), 2).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


# ### 1.1 - Create placeholders
# 
# TensorFlow requires that you create placeholders for the input data that will be fed into the model when running the session.
# 
 
# create_placeholders

def create_placeholders(n_H0, n_W0, n_y):

    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input data (number of brain regions)
    n_W0 -- scalar, width of an input image (number of time points in the input time series)
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (˜2 lines)
    #X = tf.placeholder(tf.float32, shape= (None, n_H0, n_W0, n_C0))
    X = tf.placeholder(tf.float32, shape= (None, n_H0, n_W0)) # 21_5 CORRECT INIT to (,90,300)
    Y = tf.placeholder(tf.float32, shape= (None, n_y))
    ### END CODE HERE ###

    return X, Y


#initialize_parameters

# note that this just provides the specifications to tf, the actual init happens only when you do global_variables init
def initialize_parameters():
    """
    Initializes weight parameters to build a DTM neural network with tensorflow. The shapes are:
                        W: [1,90,90] the implicit transform layer of the DTM
                        W1 : [4, 4, 3, 8] [filter h-w, inputcha, outputcha]- this will change as we modify
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1) # set the tensorflow seed
        
    
    # CHANGE 19_5: Adding the DTM hidden layer W
    # CAN add the W (90 x 90) - make it of dimension 90 x 90 and then 
    # repeat to batch size where required
    
    # 25_5 W is initialized randomly, the implicit transform captured by the weights of the DTM(W) eventually performs much
    # better than the SFM transformation (see paper for details)
    W_load = tf.get_variable("W_load", [90,90], initializer = tf.contrib.layers.xavier_initializer(seed = 0)) 
    
    
    ######### specifying the weights of the DTM ##################
    W= tf.expand_dims(W_load,axis=0) #  expanded to 1,90,90
    W1 = tf.get_variable("W1", [4,4,1,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0)) 
    W2 = tf.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    ########################
    
    # return the weights as a parameter dictionary for later use
    parameters = {"W1": W1,
                  "W2": W2,
                  "W":W}
    
    return parameters


# forward_propagation

def forward_propagation(X, parameters,l2_weight,dropout):
    """
    Implements the forward propagation for the model:
   -(???)-> implicit transform W -> CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # put here as called only once during graph setup    
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2_weight) # CHANGE 8_4, coz of api can add directly on to fc.
    
    

    # Retrieve the parameters from the dictionary "parameters" 
    W = parameters['W'] 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    
    ############################
    # use the implicit transform W to get the 90 x 90 cross correlation matrices 
    
    # 19_5 ADD the matrix operations here - to conv 90x tn batch to 90x90 batch,
    # need rep of W matrix 
    
    # SEE FORMULA FROM PAPER
    ##################
    
    
    W= tf.tile(W, [tf.shape(X)[0],1,1])  # 1,90,90 gives ntrain, 90,90
    
# imp: just append W on both sides to loaded xcorr . coz of linear property, we get trans of xcorr
# rationale: since W is linear the transformation applied to cross correlations of X is the same as the cross correlations
# of the transformed input (WX) [check paper for more details]
    left= tf.matmul(W,X) # argument is that this is xcorr precomputa
    X= tf.matmul(left,W,transpose_b=True) # 25_5 does WcorrW_T 
    
    
    X= tf.expand_dims(X,axis=3) # ntrain,90,90,1, this is the correlation matrix of the transfomed X ie Y= WX 
    
    
 ############################# forward propogation of the proposed DTM #############################################33   
    # CONV2D: stride of 1, padding 'SAME'

#tf.nn.leaky_relu(features,alpha=0.2) 
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1= tf.nn.leaky_relu(Z1,alpha=0.2) 
    #A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME') # change P should be conv now
    # RELU
    A2= tf.nn.leaky_relu(Z2,alpha=0.2)
    #A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function as required by tf api
    # 2 neurons in output layer representing the class probabilities P(Y=1|X) and P(Y=0|X).
    Z3 = tf.contrib.layers.fully_connected(P2, 2,activation_fn=None, weights_regularizer=regularizer)
    Z3_drop = tf.contrib.layers.dropout(Z3,dropout) # adding dropout and returning that, CHANGE 8/4
    
    ### END CODE HERE ###
 
   

    return X, Z3_drop # CHANGE 22_5 returning the corr mats for sep cost computation 


##########################################################:



# Compute cost 

# pass in parameters to compute l2 loss
def compute_cost(Z3, Y, parameters,l2_weight,C,lsep_weight): 

    # CHANGE 22_5 need to pass in xcorr compu for Lsep calc
    # Can use y for getting class indices in the random batch
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    W1 = parameters['W1']
    W2 = parameters['W2'] # later also retrieve fully connected weights       

    ### START CODE HERE ### (1 line of code)
    #cost_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    cost_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)) 
   

    ##############################

# Optional code to compute an explicit seperability cost, weighted by lsep_weight, not used
# (ie lsep_weight= 0) eventually.

# 21_5 add the L_sep computations here
# C has dims (371,90,90,1)

# reduce_mean and ind by a dimension 
    labels= tf.argmax(Y, 1) # just the label 1-d tensor
    labels= tf.transpose(labels) # to make sure it is n,1

    condition= tf.equal(labels, 1)
    adhd_indices = tf.where( condition )[:,0]
    C_sub_adhd= tf.gather(C,adhd_indices,axis=0) # the adhd sub tensor

    tdc_indices = tf.where( tf.equal(labels, 0) )[:,0]
    C_sub_tdc= tf.gather(C,tdc_indices,axis=0) # the adhd sub tensor


    mean_adhd = tf.reduce_mean(C_sub_adhd, axis=0, keep_dims=True)
    mean_tdc = tf.reduce_mean(C_sub_tdc, axis=0, keep_dims=True)

    lsep_cost= 1/ ( tf.norm( tf.subtract(mean_adhd,mean_tdc) , ord='fro', axis=[-3,-2] ) ) #imp- shape of mean was 
    print (lsep_cost.shape)    
    #################

    
    ##############################
# CHANGE 22_5 compute total cost using the cost_train, lsep_cost and l2_loss
    cost= cost_train + lsep_weight*lsep_cost + l2_weight*(sum(tf.losses.get_regularization_losses())) + l2_weight*tf.nn.l2_loss(W1) + l2_weight*tf.nn.l2_loss(W2) # add the two losses and divide by the number of 2*training samples ,DOUBT why no l2_weight on the function call? CHANGE 24/4 - added l2_weight to 
# func call as well.
    ### END CODE HERE ###
    
    return cost


# put the DTM model together and perform the learning using the training input
# hyperparameters are passed in as arguments to the function
def model(run_num,X_train, Y_train, X_test, Y_test, learning_rate = 0.001,num_epochs = 100, minibatch_size = 32, print_cost = True, l2_weight=0.001, lsep_weight=0,dropout=1): # dropout is actually the drop-out keep probability # l2_wright - 0.009, dropout - 1.0 
    """
    
    
    Arguments:
    X_train -- training set, of shape (None, 90, T_n, 1)
    Y_train -- test set, of shape (None, n_y = 2)
    X_test -- training set, of shape (None, 90, T_n, 1)
    Y_test -- test set, of shape (None, n_y = 2)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    print('...................')
    print ("Run_num:",run_num)
   # print (run_num)    
    print('...................')

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    #(m, n_H0, n_W0, n_C0) = X_train.shape
    (m, n_H0, n_W0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    #X, Y = create_placeholders(n_H0=n_H0,n_W0=n_W0,n_C0=n_C0,n_y=n_y)
    X, Y = create_placeholders(n_H0=n_H0,n_W0=n_W0,n_y=n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)

    #22_5 C is the tensor containing the correlations for the current minibatch data
    C,Z3 = forward_propagation(X, parameters,l2_weight,dropout) # not calc at once in vectorized fashion, 

    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3,Y,parameters, l2_weight,C,lsep_weight)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        #print('................................')
        #print('Counting params') # CHANGE 26_4- Adding number of parameters
        #print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
       # print('................................')
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        print('................................')
        print("parameter_count =", sess.run(parameter_count))
        print('................................')
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1 # do we have to explicitly change the seed ourselves?
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed) # very nice method

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost 5 epochs
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1) # the arg 1 is for axis, the index of max is assumed to be pred class
        ylabel_op= tf.argmax(Y, 1)
        correct_prediction = tf.equal(predict_op, ylabel_op) # the argmax here is to conv 1 hot back to label
        # how to extract predict_op (y_hat) and ylabel_op(y_true)
        # Calculate accuracies
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # this defines the accuracy graph operations
        #print(accuracy)
        
        # finally after training - get the pred, true y and the accuracy
        print('Calcu the accuracies...')
        #train_accuracy = accuracy.eval({X: X_train, Y: Y_train}) # even this is a graph operation, check if getting printed
        
        test_result_vals= sess.run([accuracy,ylabel_op,predict_op], feed_dict={X: X_test, Y: Y_test}) # this works, thats why it is called fetches
        
        test_accuracy= test_result_vals[0]
        print(str(run_num)+ " Test Accuracy:", test_accuracy)
        
        
        
        ####################################
        # get test confu mat and other metrics
        test_confu= confusion_matrix(test_result_vals[1], test_result_vals[2])
        preci= precision_score(test_result_vals[1], test_result_vals[2])
        sensi= recall_score(test_result_vals[1], test_result_vals[2])
        f1= f1_score(test_result_vals[1], test_result_vals[2])
        
        tn, fp, fn, tp = confusion_matrix(test_result_vals[1], test_result_vals[2]).ravel()
        specifi = float(tn) / (tn+fp) 
        # print confusion matrix- CHANGE 12_4
        print ('........................')
        print (test_confu) # put in true and pred labels
        print ('........................')
        print (preci)
        print (sensi)
        print (f1)
        print (specifi)
        # save test confu and the calculated metrics in text files- named by data      
        print ('........................')


        metrics= []
        metrics.append(test_result_vals[0])
        metrics.append(preci)
        metrics.append(sensi)
        metrics.append(f1)
        metrics.append(specifi)

        metrics= np.asarray(metrics) # conv to np array
        cwd = os.getcwd()

# save at .txt files
        np.savetxt(cwd+'/Results/Runs/23_5_implicit_corr/'+dataname+'Run_'+str(run_num)+'_testmetrics.txt', metrics, delimiter=' ') # save the metrics file
        np.savetxt(cwd+'/Results/Runs/23_5_implicit_corr/'+dataname+'Run_'+str(run_num)+'_testconfu.txt', test_confu, delimiter=' ') # save the test confu mat 
        np.savetxt(cwd+'/Results/Runs/LabelnPred/'+'Run_'+str(run_num)+dataname+'_testlabel.txt', test_result_vals[1], delimiter=' ') 
        np.savetxt(cwd+'/Results/Runs/LabelnPred/'+dataname+'Run_'+str(run_num)+'_testpredic.txt', test_result_vals[2], delimiter=' ') 
        ###################################
        # save the cost data
        cost_data= np.squeeze(costs)        
        #savefilecost= dataname+'_costfile.h5'
        #h5f = h5py.File(cwd+'/Results/Runs/'+'Run_'+str(run_num)+savefilecost, 'w')
        #h5f.create_dataset('costarr', data=cost_data)
        #h5f.close()

        # save the image for the cost data
        plt.plot(cost_data)
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig(cwd+'/Results/Runs/23_5_implicit_corr/'+dataname+'Run_'+str(run_num)+'_learning_curve.png',dpi=600) # check
        #plt.show()
        

        ###################################
        #compute metrics on train data
                
 
        train_result_vals= sess.run([accuracy,ylabel_op,predict_op], feed_dict={X: X_train, Y: Y_train}) # this works, thats why it is called fetches
        train_accuracy= train_result_vals[0]         
        print(str(run_num)+" train Accuracy:", train_accuracy)
        

        train_confu= confusion_matrix(train_result_vals[1], train_result_vals[2])
        preci= precision_score(train_result_vals[1], train_result_vals[2])
        sensi= recall_score(train_result_vals[1], train_result_vals[2])
        f1= f1_score(train_result_vals[1], train_result_vals[2])
        
        tn, fp, fn, tp = confusion_matrix(train_result_vals[1], train_result_vals[2]).ravel()
        specifi = float(tn) / (tn+fp) 
        # print confusion matrix- CHANGE 12_4
        print ('........................')
        print (train_confu) # put in true and pred labels
        print ('........................')
        print (preci)
        print (sensi)
        print (f1)
        print (specifi)
        # save test confu and the calculated metrics in text files- named by data, check if working and then move to train confu      
        print ('........................')

        metrics= []
        metrics.append(train_result_vals[0])
        metrics.append(preci)
        metrics.append(sensi)
        metrics.append(f1)
        metrics.append(specifi)

        metrics= np.asarray(metrics) # conv to np array
        cwd = os.getcwd()

        np.savetxt(cwd+'/Results/Runs/23_5_implicit_corr/'+dataname+'Run_'+str(run_num)+ '_trainmetrics.txt', metrics, delimiter=' ') # save the metrics file
        np.savetxt(cwd+'/Results/Runs/23_5_implicit_corr/'+dataname+'Run_'+str(run_num)+'_trainconfu.txt', train_confu, delimiter=' ') 
        np.savetxt(cwd+'/Results/Runs/LabelnPred/'+'Run_'+str(run_num)+dataname+'_trainlabel.txt', train_result_vals[1], delimiter=' ') 
        np.savetxt(cwd+'/Results/Runs/LabelnPred/'+dataname+'Run_'+str(run_num)+'_trainpredic.txt', train_result_vals[2], delimiter=' ') 

        return train_accuracy, test_accuracy, parameters
#######################################################


# 4_5 just add a loop here, and put run number as an argument

total_runs= 25

train_accu_list= []
test_accu_list= [] # find max index of test accu

for run_num in range(0,total_runs):
    curr_train_acc, curr_test_acc, parameters = model(run_num,X_train, Y_train, X_test, Y_test) 
    
    train_accu_list.append(curr_train_acc)
    test_accu_list.append(curr_test_acc)

# perform global optimization using simultaneuos searches
print('....................')
print('printing all test accus')
print (test_accu_list)
print('....................')

max_test_acc = max(test_accu_list)
max_test_acc_index = test_accu_list.index(max_test_acc)
print ('max test acc: ', max_test_acc)
print ('max test acc run index: ', max_test_acc_index)
# find max train acc and the run number
max_train_acc = max(train_accu_list)
max_train_acc_index = train_accu_list.index(max_train_acc)
print ('max train acc: ', max_train_acc)

print('................')
print ('avg test accuracy: ', sum(test_accu_list) / float(len(test_accu_list)) )
print ('avg train accuracy: ', sum(train_accu_list) / float(len(train_accu_list)) )

print('................')

######## also save the hyperparameters
#hyps= []
#hyps.append(learning_rate)
#learning_rate = 0.001,num_epochs = 70, minibatch_size = 32, print_cost = True, l2_weight=0.001, lsep_weight=0.00,dropout=1

# In[14]: