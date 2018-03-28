import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import sys

def local():
    
    img_rows,img_cols = 28,28



    sign = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    str = "imageconv/gray.jpeg"

    gray = Image.open(str)
    im1 = np.array(gray)



    test = pd.read_csv('datasets/sign-language-mnist/sign_mnist_test.csv')


    im2 = im1.reshape(1,28,28,1)

    test_data = test.iloc[:1,1:].values.reshape(1,28,28,1)
    print im2


    print type(test_data),type(im1)


    model1 = load_model('my_model.h5')

    model1.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])


    prict = model1.predict(im2)

    sol = np.argmax(prict,axis=1)

    m  = sol[0]
    print sign[m]
    return sign[m]
if __name__ == '__main__':
   local()

                                                                                                                                                                                  
                           
