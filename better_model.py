import os
# import os
import IPython
import cv2 as cv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import keras
# from keras import layers
# from keras.layers import concatenate
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical , plot_model
from tensorflow.keras.layers import Input, Add,Average, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, SeparableConv2D, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,concatenate,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
 

DIR='chest_xray/train'

def extract_data(dir_path):
  classes=os.listdir(dir_path)
  Features=[]
  Labels=[]
  
  if dir_path == 'chest_xray/train':
      for c in classes:
        if c =='PNEUMONIA':
            path= os.path.join(dir_path,c)
            images= os.listdir(path)
            images = images[2400:] #stating 1500 images 
            print(f"Class ----------> {c}")
            for idx ,image in tqdm(enumerate(images) ,total=len(images)):
              image_path=os.path.join(path,image)
              img= cv.imread(image_path)
              # gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
              image = cv.resize(img, (64, 64))
              Features.append(image)
              Labels.append(c)
        else:
            path= os.path.join(dir_path,c)
            images= os.listdir(path)
            print(f"Class ----------> {c}")
            for idx ,image in tqdm(enumerate(images) ,total=len(images)):
              image_path=os.path.join(path,image)
              img= cv.imread(image_path)
              # gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
              image = cv.resize(img, (64, 64))
              Features.append(image)
              Labels.append(c)
  else:
     for c in classes:
         path= os.path.join(dir_path,c)
         images= os.listdir(path)
         images = images[:1500] #stating 1500 images 
         print(f"Class ----------> {c}")
         for idx ,image in tqdm(enumerate(images) ,total=len(images)):
           image_path=os.path.join(path,image)
           img= cv.imread(image_path)
           # gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
           image = cv.resize(img, (64, 64))
           Features.append(image)
           Labels.append(c)
                
  Features=np.array(Features)
  # Labels=to_catagorical(Labels,num_classes= len(classes))
  return Labels , Features ,classes

# from google.colab import drive
# drive.mount('/content/drive')

Y_train, X_train ,classes = extract_data(DIR)
DIR_ = 'chest_xray/test'
Y_test, X_test , classes=extract_data(DIR_)

RES=[]
for i in Y_train:
  if i == "PNEUMONIA":
    RES.append([0,0])
  else:
    RES.append([1,0])

RES=np.array(RES)


Y_train=RES

REST=[]
for i in Y_test:
  if i == "PNEUMONIA":
    REST.append([0,0])
  else:
    REST.append([1,0])

REST=np.array(REST)


Y_test=REST

print(" X-train",X_train.shape)
print(" Y-train",Y_train.shape)
print(" X-test",X_test.shape)
print(" Y-test",Y_test.shape)


def identity_block(X, f, filters, stage, block):

  '''
  Implementation of identity block described above

  Arguments:
  X -       input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
  f -       defines shpae of filter in the middle layer of the main path
  filters - list of integers, defining the number of filters in each layer of the main path
  stage -   defines the block position in the network
  block -   used for naming convention

  Returns:
  X - output is a tensor of shape (n_H, n_W, n_C) which matches (m, n_H_prev, n_W_prev, n_C_prev)
  '''

  # defining base name for block
  conv_base_name = 'res' + str(stage) + block + '_'
  bn_base_name = 'bn' + str(stage) + block + '_'

  # retrieve number of filters in each layer of main path
  # NOTE: f3 must be equal to n_C. That way dimensions of the third component will match the dimension of original input to identity block
  f1, f2, f3 = filters

  # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
  bn_axis = 3

  # save input for "addition" to last layer output; step in skip-connection
  X_skip_connection = X

  # ----------------------------------------------------------------------
  # Building layers/component of identity block using Keras functional API

  # First component/layer of main path
  X = Conv2D(filters= f1, kernel_size = (1,1), strides = (1,1), padding='valid', name=conv_base_name+'first_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'first_component')(X)
  X = Activation('relu')(X)

  # Second component/layer of main path
  X = Conv2D(filters= f2, kernel_size = (f,f), strides = (1,1), padding='same', name=conv_base_name+'second_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'second_component')(X)
  X = Activation('relu')(X)

  # Third component/layer of main path
  X = Conv2D(filters= f3, kernel_size = (1,1), strides = (1,1), padding='valid', name=conv_base_name+'third_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'third_component')(X)

  # "Addition step" - skip-connection value merges with main path
  # NOTE: both values have same dimensions at this point, so no operation is required to match dimensions
  X = Add()([X, X_skip_connection])
  X = Activation('relu')(X)

  return X

def identity_block_S(X, f, filters, stage, block):

  '''
  Implementation of identity block described above

  Arguments:
  X -       input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
  f -       defines shpae of filter in the middle layer of the main path
  filters - list of integers, defining the number of filters in each layer of the main path
  stage -   defines the block position in the network
  block -   used for naming convention

  Returns:
  X - output is a tensor of shape (n_H, n_W, n_C) which matches (m, n_H_prev, n_W_prev, n_C_prev)
  '''

  # defining base name for block
  conv_base_name = 'res' + str(stage) + block + '_'
  bn_base_name = 'bn' + str(stage) + block + '_'

  # retrieve number of filters in each layer of main path
  # NOTE: f3 must be equal to n_C. That way dimensions of the third component will match the dimension of original input to identity block
  f1, f2, f3 = filters

  # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
  bn_axis = 3

  # save input for "addition" to last layer output; step in skip-connection
  X_skip_connection = X

  # ----------------------------------------------------------------------
  # Building layers/component of identity block using Keras functional API

  # First component/layer of main path
  X = Conv2D(filters= f1, kernel_size = (1,1), strides = (1,1), padding='valid', name=conv_base_name+'first_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'first_component')(X)
  X = Activation('relu')(X)

  # Second component/layer of main path
  X = SeparableConv2D(filters= f2, kernel_size = (f,f), strides = (1,1), padding='same', name=conv_base_name+'second_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'second_component')(X)
  X = Activation('relu')(X)

  # Third component/layer of main path
  X = Conv2D(filters= f3, kernel_size = (1,1), strides = (1,1), padding='valid', name=conv_base_name+'third_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'third_component')(X)

  # "Addition step" - skip-connection value merges with main path
  # NOTE: both values have same dimensions at this point, so no operation is required to match dimensions
  X = Add()([X, X_skip_connection])
  X = Activation('relu')(X)

  return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in above figure

    Arguments:
    X -       input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -       defines shape of filter in the middle layer of the main path
    filters - list of integers, defining the number of filters in each layer of the main path
    stage -   defines the block position in the network
    block -   used for naming convention
    s -       specifies the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining base name for block
    conv_base_name = 'res' + str(stage) + block + '_'
    bn_base_name = 'bn' + str(stage) + block + '_'

    # retrieve number of filters in each layer of main path
    f1, f2, f3 = filters

    # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
    bn_axis = 3

    # save input for "addition" to last layer output; step in skip-connection
    X_skip_connection = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(f1, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'first_component', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = bn_axis, name = bn_base_name + 'first_component')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(f2,  kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_base_name + 'second_component', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = bn_axis, name = bn_base_name + 'second_component')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(f3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_base_name + 'third_component', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = bn_axis, name = bn_base_name + 'third_component')(X)

    ##### Convolve skip-connection value to match its dimensions to third layer output's dimensions ####
    X_skip_connection = Conv2D(f3, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'merge', kernel_initializer = glorot_uniform(seed=0))(X_skip_connection)
    X_skip_connection = BatchNormalization(axis = 3, name = bn_base_name + 'merge')(X_skip_connection)

    # "Addition step"
    # NOTE: both values have same dimensions at this point
    X = Add()([X, X_skip_connection])
    X = Activation('relu')(X)

    return X

def convolutional_block_S(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in above figure

    Arguments:
    X -       input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -       defines shape of filter in the middle layer of the main path
    filters - list of integers, defining the number of filters in each layer of the main path
    stage -   defines the block position in the network
    block -   used for naming convention
    s -       specifies the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining base name for block
    conv_base_name = 'res' + str(stage) + block + '_'
    bn_base_name = 'bn' + str(stage) + block + '_'

    # retrieve number of filters in each layer of main path
    f1, f2, f3 = filters

    # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
    bn_axis = 3

    # save input for "addition" to last layer output; step in skip-connection
    X_skip_connection = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(f1, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'first_component', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = bn_axis, name = bn_base_name + 'first_component')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = SeparableConv2D(f2,  kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_base_name + 'second_component', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = bn_axis, name = bn_base_name + 'second_component')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(f3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_base_name + 'third_component', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = bn_axis, name = bn_base_name + 'third_component')(X)

    ##### Convolve skip-connection value to match its dimensions to third layer output's dimensions ####
    X_skip_connection = Conv2D(f3, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'merge', kernel_initializer = glorot_uniform(seed=0))(X_skip_connection)
    X_skip_connection = BatchNormalization(axis = 3, name = bn_base_name + 'merge')(X_skip_connection)

    # "Addition step"
    # NOTE: both values have same dimensions at this point
    X = Add()([X, X_skip_connection])
    X = Activation('relu')(X)

    return X

def gnt(input_shape = (64, 64, 3), classes = 2):
    """
    Arguments:
    input_shape - shape of the images of the dataset
    classes - number of classes

    Returns:
    model - a Model() instance in Keras

    """

    # plug in input_shape to define the input tensor
    X_input = Input(input_shape)

    # Zero-Padding : pads the input with a pad of (3,3)
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv_1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # NOTE: dimensions of filters that are passed to identity block are such that final layer output
    # in identity block mathces the original input to the block
    # blocks in each stage are alphabetically sequenced

    Y = X

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    #stage 2.1
    Y= convolutional_block_S(Y, f = 3, filters = [64, 64, 256], stage = 2.1, block='a', s = 1)
    Y = identity_block_S(Y, 3, [64, 64, 256], stage=2.1, block='b')
    Y = identity_block_S(Y, 3, [64, 64, 256], stage=2.1, block='c')

    X = Average()([X, Y])
    X1 = Conv2D(256, (3, 3), strides = (1,1), padding = 'valid', name ='merge1', kernel_initializer = glorot_uniform(seed=0))(X)
    X1 = BatchNormalization(axis = 3, name = 'bn_1_1')(X1)
    # X1 = Activation('relu')(X1)
    # X1 = Add()([X,X1])
    X1 = MaxPooling2D((3, 3), strides=(2, 2))(X1)
    X1 = Flatten()(X1)
    # X1 = Dense(1024, activation='relu', name='fc1.0' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X1)
    # X1= Dropout(0.5)(X1)
    # X1 = Dense(classes, activation='softmax', name='fc1.11' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X1)
    X1 = Dense(classes, activation='softmax', name='fc1.1' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X1)

    Y=X
    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 3.1
    Y = convolutional_block_S(Y, f=3, filters=[128, 128, 512], stage=3.1, block='a', s=2)
    Y = identity_block_S(Y, 3, [128, 128, 512], stage=3.1, block='b')
    Y = identity_block_S(Y, 3, [128, 128, 512], stage=3.1, block='c')
    Y = identity_block_S(Y, 3, [128, 128, 512], stage=3.1, block='d')

    X = Average()([X, Y])
    X2 = Conv2D(256, (3, 3), strides = (2,2), padding = 'valid', name ='merge1', kernel_initializer = glorot_uniform(seed=0))(X)
    X2 = BatchNormalization(axis = 3, name = 'bn_1_1')(X)
    # X2 = Activation('relu')(X2)
    X2 = MaxPooling2D((3, 3))(X)
    X2 = Flatten()(X2)
    # X2 = Dense(1024, activation='relu', name='fc1.2' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X2)
    # # X2= Dropout(0.7)(X2)
    X2 = Dense(classes, activation='softmax', name='fc1.3' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X2)
    
    Y=X
    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    #stage 4.1
    Y = convolutional_block_S(Y, f=3, filters=[256, 256, 1024], stage=4.1, block='a', s=2)
    Y = identity_block_S(Y, 3, [256, 256, 1024], stage=4.1, block='b')
    Y = identity_block_S(Y, 3, [256, 256, 1024], stage=4.1, block='c')
    Y = identity_block_S(Y, 3, [256, 256, 1024], stage=4.1, block='d')
    Y = identity_block_S(Y, 3, [256, 256, 1024], stage=4.1, block='e')
    Y = identity_block_S(Y, 3, [256, 256, 1024], stage=4.1, block='f')

    X = Average()([X, Y])
    X3 = BatchNormalization(axis = 3, name = 'bn_1_1')(X)
    X3 = MaxPooling2D((3, 3), strides=(2, 2))(X3)
    X3 = Flatten()(X3)
    # X3 = Dense(1024, activation='softmax', name='fc1.0' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X1)
    # X3= Dropout(0.7)(X3)
    X3 = Dense(classes, activation='softmax', name='fc1.5' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X3)

    Y=X
    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # Stage 5.1
    Y = convolutional_block_S(Y, f=3, filters=[512, 512, 2048], stage=5.1, block='a', s=2)
    Y = identity_block_S(Y, 3, [512, 512, 2048], stage=5.1, block='b')
    Y = identity_block_S(Y, 3, [512, 512, 2048], stage=5.1, block='c')

    X = Average()([X, Y])
    X = BatchNormalization(axis = 3, name = 'bn_1_1')(X)
    # Average Pooling
    X = AveragePooling2D((2, 2), name='avg_pool_1')(X)

    # Y = AveragePooling2D((2, 2), name='avg_pool_2')(Y)

    # output layer
    X = Flatten()(X)
    # Y = Flatten()(Y)

    # Y = concatenate([X,Y],axis=-1)
    # Y = Dense(1000, activation='relu', name='fc1000', kernel_initializer = glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.03))(Y)
    # Y = Dropout(0.5)(Y)
    # Y = Dense(128, activation='relu', name='fc128', kernel_initializer = glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.03))(Y)
    # Y = Dropout(0.5)(Y)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
# kernel_regularizer=tf.keras.regularizers.l2(0.03)
    # Create model
    model = Model(inputs = X_input, outputs = [X], name='ResNet50')

    return model

def ResNet50(input_shape = (64, 64, 3), classes = 2):
    """
    Arguments:
    input_shape - shape of the images of the dataset
    classes - number of classes

    Returns:
    model - a Model() instance in Keras

    """

    # plug in input_shape to define the input tensor
    X_input = Input(input_shape)

    # Zero-Padding : pads the input with a pad of (3,3)
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv_1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # NOTE: dimensions of filters that are passed to identity block are such that final layer output
    # in identity block mathces the original input to the block
    # blocks in each stage are alphabetically sequenced

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # Average Pooling
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

datagen = ImageDataGenerator(
    rotation_range=40,  # Rotate images randomly up to 40 degrees
    width_shift_range=0.2,  # Shift images horizontally by up to 20%
    height_shift_range=0.2,  # Shift images vertically by up to 20%
    shear_range=0.2,  # Apply shear transformation
    zoom_range=0.2,  # Zoom images randomly by up to 20%
    horizontal_flip=False,  # Flip images horizontally randomly
    fill_mode='nearest'  # Fill empty pixels with nearest neighbour value
)

# adam = Adam(learning_rate=0.01)
# model = ResNet50(input_shape = (64, 64, 3), classes = 2)
# # model.summary()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
# # IPython.display.Image('gnt.png')

# # Define a dictionary to store losses for each batch size
# batch_losses = []

# def on_train_batch_end(batch, logs=None):
#     batch_losses.append(logs['loss'])

# # Create a callback instance and add it to your training
# # callback = tf.keras.callbacks.LambdaCallback(on_train_batch_end=on_train_batch_end)

# # model.fit(X_train,Y_train, epochs=10, batch_size=32)  # Include validation data
# # Create a callback instance and add it to your training
# callback = tf.keras.callbacks.LambdaCallback(on_train_batch_end=on_train_batch_end)
# # model.fit(X_train, Y_train, epochs=4, batch_size=32, callbacks=[callback])
# model.fit(
#     datagen.flow(X_train, Y_train, batch_size=32),  # Use data generator for training
#     epochs=10,
#     validation_data=(X_test, Y_test),
#     callbacks=[callback]
#     )

adam = Adam(learning_rate=0.01)
model = gnt(input_shape = (64, 64, 3), classes = 2)
# model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# plot_model(model, to_file='mgnt.png', show_shapes=True)
# IPython.display.Image('gnt.png')

# Define a dictionary to store losses for each batch size
batch_losses = []

def on_train_batch_end(batch, logs=None):
    batch_losses.append(logs['loss'])

# Create a callback instance and add it to your training
callback = tf.keras.callbacks.LambdaCallback(on_train_batch_end=on_train_batch_end)
model.fit(X_train, Y_train, epochs=18, batch_size=32, callbacks=[callback])

model_1 = ResNet50(input_shape = (64, 64, 3), classes = 2)
# model_1.summary()
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
batch_losses_1 = []

def on_train_batch_end_1(batch, logs=None):
    batch_losses_1.append(logs['loss'])

# Create a callback instance and add it to your training
callback = tf.keras.callbacks.LambdaCallback(on_train_batch_end=on_train_batch_end_1)
model_1.fit(X_train, Y_train, epochs=18, batch_size=32, callbacks=[callback])

# Since batch size is fixed, plot the loss values directly
plt.plot(batch_losses,label='model1(1) new')
plt.plot(batch_losses_1,label='Resnet model')
plt.xlabel('Iteration (within batch size)')
plt.ylabel('Loss')
plt.title('Loss vs. Batch Size')
plt.legend()
plt.show()
plt.savefig('my_plot_new_1(1).png')

predictions = model.evaluate(X_test, Y_test)
print("Loss = " + str(predictions[0]))
print("Test Accuracy = " + str(predictions[1]))

predictions_1 = model_1.evaluate(X_test, Y_test)
print("Loss = " + str(predictions_1[0]))
print("Test Accuracy = " + str(predictions_1[1]))
