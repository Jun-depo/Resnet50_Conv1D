import tensorflow as tf

import keras
from keras.engine.input_layer import Input
import keras.backend as K

from keras.utils import plot_model, to_categorical
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D, BatchNormalization

from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 

from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.engine.topology import Layer

def identity_block(X, f, filter_numbers, stage, block):
    """
    Implementation of the identity block as defined in Figure 4.
    
    Arguments:
    X -- input tensor of shape (m, input_length_prev (n_w), input_Channel_prev (n_c).
    f -- kernel_size, integer, shape of convolution filter in the main path.
    filter_numbers -- python list of integers, defining the number of filters in the CONV layers of the main path.
    stage -- integer, used to name the layers, depending on their position in the network.
    block -- string/character, used to name the layers, depending on their position in the network.
    
    Returns:
    X -- output of the identity block, tensor of shape (n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filter_numbers
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First Conv layer
    X = Conv1D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 2, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
   
    
    # Second Conv layer
    X = Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 2, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third Conv layer
    X = Conv1D(filters = F3, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 2, name = bn_name_base + '2c')(X)

    # Merge with Residual shortcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, input_length_prev (n_w), input_Channel_prev (n_c))
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv1D(F1, 1, strides = s, name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 2, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv1D(F2, f, strides = 1, padding = "same", name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 2, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv1D(F3, 1, strides = 1, name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 2, name = bn_name_base + '2c')(X)
    
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv1D(F3, 1, strides = s, name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    #X_shortcut = BatchNormalization(axis = 2, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def ResNet50(input_shape = (30000, 1), n_out=1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV1D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the 1D data
    n_out -- integer, number of classes or output

    Returns:
    model -- a Model() instance in Keras

    params here were used in one of my projects. 
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = MaxPooling1D(10, strides=5)(X_input) 
    
    # Zero-Padding
    X = ZeroPadding1D(3)(X)
    
    # stage 1, 64 filters, kernel_size=7
    X = Conv1D(64, 7, strides = 2, name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 2, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, strides=2)(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [16, 16, 64], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [16, 16, 64], stage=2, block='b')
    X = identity_block(X, 3, [16, 16, 64], stage=2, block='c')

    ### START CODE HERE ###
    
    X = convolutional_block(X, f = 3, filters = [32,32,128], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [32,32,128], stage=3, block='b')
    X = identity_block(X, 3, [32,32,128], stage=3, block='c')
    X = identity_block(X, 3, [32,32,128], stage=3, block='d')

    # Stage 4 (≈6 lines)
    
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='c')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='d')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='e')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='f')

    # Stage 5 
   
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=5, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=5, block='c')

    X = AveragePooling1D(pool_size=5)(X)
    
    # output layer
    X = Flatten()(X)
    X = Dropout(0.5)(X)
    
    # For regression
    X = Dense(n_out, name='fc-dense', kernel_initializer = glorot_uniform(seed=0), 
              kernel_regularizer=regularizers.l2(0.2), bias_regularizer=regularizers.l2(0.2))(X)
    
    # for classification, if n_out =1, add:  
    # X = Activation('sigmoid')(X)

    # if n_out > 1, add:  
    # X = Activation('softmax')(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50_1d')

    return model
