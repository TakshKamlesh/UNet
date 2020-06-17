#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2
import numpy as np
import os


# In[ ]:


print("Enter image path")
path = input()
print("Choose from the following trained models")
model_list = os.listdir('Models')
for m in model_list:
    print(m)
model_name = input()


# In[ ]:


model = tf.keras.models.load_model('Models/' + model_name)


# In[ ]:


image = cv2.imread(path)
predictions = model.predict(np.array([image]))


# In[ ]:


cv2.imshow(predictions[0])


# In[ ]:


cv2.imwrite('Predictions/' + predictions[0])

