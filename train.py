#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import cv2
from tqdm import tqdm
import os     
import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[ ]:


print("Enter data path")
path = input()


# In[1]:


def load_data(path):
  
    x = []
    y = []
    images = os.path.join(path , 'images')
    masks = os.path.join(path , 'masks')
    image_folder = os.listdir(images)
    for i in image_folder:
        
        ci = cv2.imread(images +'/' + i)
        # ci = cv2.cvtColor(ci , cv2.COLOR_BGR2GRAY)
        # ci = np.reshape(ci , (224,224,1))
        cm = cv2.imread(masks + '/' + i)
        cm = cm[:,:,0]
        x.append(ci)
        y.append(cm)
        

    return np.array(x) , np.array(y)


# In[ ]:


X , Y = load_data()


# In[ ]:


X.shape


# In[2]:


def vgg_encoder(trainable = False , weights = 'imagenet', input_shape = (224,224,3)):

    vgg_encoder = tf.keras.applications.vgg16.VGG16(weights = weights , include_top = False,input_shape = input_shape)

    if not trainable:

        for layer in vgg_encoder.layers:

            layer.trainable = False

    return vgg_encoder


# In[ ]:


te = vgg_encoder()
te.summary()


# In[3]:


def vgg_decoder():

    decoder_input = tf.keras.layers.Input(shape = (None,None,512))

    x = tf.keras.layers.Conv2D(filters = 512 , kernel_size = (3,3) , padding = 'same' , activation = 'relu')(decoder_input)
    #  print(x.shape)
    x = tf.keras.layers.Conv2DTranspose(filters = 512 , kernel_size=(3,3) , padding = 'same' , strides = (2,2) , activation='relu')(x)
    #  print(x.shape)
    x = tf.keras.layers.Conv2D(filters = 512 , kernel_size = (3,3) , padding = 'same' , activation='relu')(x)
    #  x = tf.keras.layers.Concatenate(axis=0)([encoder.get_layer('block5_conv3').output , x])
    x = tf.keras.layers.Conv2D(filters = 512 , kernel_size = (3,3) , padding = 'same' , activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters = 512 , kernel_size = (3,3) , padding = 'same' , activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 512 , kernel_size=(3,3) ,padding = 'same' , strides = (2,2) , activation = 'relu')(x)
    #  print(x.shape)
    x = tf.keras.layers.Conv2D(filters = 512 , kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    #  x = tf.keras.layers.Concatenate()([x , encoder.get_layer('block4_conv3').output])
    x = tf.keras.layers.Conv2D(filters = 512 , kernel_size=(3,3) , padding = 'same' , activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(filters = 512 , kernel_size=(3,3) , padding = 'same' , activation = 'relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 256 , kernel_size = (3,3) , padding = 'same' ,strides = (2,2) , activation = 'relu')(x)
    #  print(x.shape)
    x = tf.keras.layers.Conv2D(filters= 256, kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    #  x = tf.keras.layers.Concatenate()([x , encoder.get_layer('block3_conv3').output])
    x = tf.keras.layers.Conv2D(filters= 256, kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(filters= 256, kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 128 , kernel_size = (3,3) ,padding = 'same' , strides = (2,2) , activation = 'relu')(x)
    #  print(x.shape)
    x = tf.keras.layers.Conv2D(filters= 128, kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    #  x = tf.keras.layers.Concatenate()([x , encoder.get_layer('block2_conv2').output])
    x = tf.keras.layers.Conv2D(filters= 128, kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(filters= 128, kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 64 , kernel_size = (3,3) , padding = 'same' ,strides = (2,2) , activation = 'relu')(x)
    #  print(x.shape)
    x = tf.keras.layers.Conv2D(filters= 64, kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    #  x = tf.keras.layers.Concatenate()([x , encoder.get_layer('block1_conv2').output])
    x = tf.keras.layers.Conv2D(filters= 64, kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    outputs = tf.keras.layers.Conv2D(filters= 1, kernel_size = (3,3) , padding = 'same' , activation = 'relu')(x)
    decoder_model = tf.keras.Model(inputs = decoder_input , outputs=outputs)

    return decoder_model


# In[4]:


def unet_generator():
   
    encoder = vgg_encoder()

    decoder = vgg_decoder()

    concat = tf.keras.layers.Concatenate()

    print(encoder.get_layer('block3conv3').output.shape , encoder)

    final_model = tf.keras.Sequential([encoder , decoder])
    return final_model


# In[ ]:


model = unet_generator()


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])


# In[ ]:


X_train, X_test , Y_train ,Y_test = train_test_split(X,Y , test_size = 0.12)


# In[ ]:


print(X_train.shape, Y_test.shape)


# In[ ]:


history = model.fit(x = X_train , y = Y_train , batch_size = 16, epochs = 300 , validation_data = (X_test,  Y_test), shuffle =True)


# In[ ]:


print("Select name for model")
name = input()
model.save('Models/' + name)
curr_dir = os.getcwd()
print("Model saved in "+curr_dir+"/Models")


# In[ ]:




