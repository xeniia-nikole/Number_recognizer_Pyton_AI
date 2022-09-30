from tensorflow.keras.datasets import mnist    
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense    
from tensorflow.keras import utils
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import BinaryCrossentropy 
from tensorflow.keras.layers import Conv2D
from tensorflow.random import normal
from tensorflow.keras import utils  
from keras.utils import np_utils
from google.colab import drive   
from os import listdir
from PIL import Image    
import numpy as np         
import pandas as pd  
import matplotlib.pyplot as plt                 
%matplotlib inline 

(x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()
x_train_org.shape
print(x_train_org[1])

# Изменение формы входных картинок с 28х28 на 784
# первая ось остается без изменения, остальные складываются в вектор
x_train = x_train_org.reshape(x_train_org.shape[0], -1)    

# Нормализация входных картинок
# Преобразование x_train в тип float32 (числа с плавающей точкой) и нормализация
x_train = x_train.astype('float32') / 255.

# Задание константы количества распознаваемых классов
CLASS_COUNT = 10

# Преобразование ответов в формат one_hot_encoding
y_train = utils.to_categorical(y_train_org, CLASS_COUNT)

model = Sequential()
model.add(Dense(800, input_dim=784, activation='relu')) 
model.add(Dense(400, activation='relu')) 
model.add(Dense(CLASS_COUNT, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
model.fit(x_train, y_train, batch_size=128, epochs=9, verbose=1)     

path = '/content/Untitled.jpg'

test_image = image.load_img(path, 
                             target_size=(28, 28), color_mode = "grayscale")

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = np.array(test_image)
test_image_r1 = test_image.reshape(test_image.shape[0], -1)  

test_image_r2 = test_image_r1 / 255.0

prediction = model.predict(test_image_r2)
pred = np.argmax(prediction)
print(f'Распознана цифра: {pred}')
