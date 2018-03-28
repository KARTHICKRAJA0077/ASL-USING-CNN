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

img_rows,img_cols = 28,28

#sign = {'[0]':}

sign = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

'''

str = raw_input("Enter the name of your text file ")
im = Image.open(str)
#img = im.resize((img_rows,img_cols))
#print img
#gray = img.convert('L')
#gray.show()
img  = im.convert('L')
gray = img.resize((img_rows,img_cols))
im1 = np.array(gray)
#print im1
'''
'''
path = raw_input("Enter the name of your text file ")


os.system("convert -type Grayscale path '/Users/diwakar/Downloads/imageconv/gray.jpeg'")

os.system("convert '/Users/diwakar/Downloads/imageconv/gray.jpeg'  -resize 28x28\!  '/Users/diwakar/Downloads/imageconv/gray.jpeg'")

os.system("convert '/Users/diwakar/Downloads/imageconv/gray.jpeg'  -filter Mitchell  '/Users/diwakar/Downloads/imageconv/gray.jpeg'")

os.system("convert '/Users/diwakar/Downloads/imageconv/gray.jpeg'  -filter Robidoux  '/Users/diwakar/Downloads/imageconv/gray.jpeg'")

os.system("convert '/Users/diwakar/Downloads/imageconv/gray.jpeg'  -filter Catrom  '/Users/diwakar/Downloads/imageconv/gray.jpeg'")

os.system("convert '/Users/diwakar/Downloads/imageconv/gray.jpeg'  -filter Spline  '/Users/diwakar/Downloads/imageconv/gray.jpeg'")

os.system("convert '/Users/diwakar/Downloads/imageconv/gray.jpeg'  -filter Hermite  '/Users/diwakar/Downloads/imageconv/gray.jpeg'")
'''
#os.system("./example.sh")
str = "/imageconv/gray.jpeg"
#raw_input("Enter the name of your text file ")
gray = Image.open(str)
im1 = np.array(gray)
#print im1


test = pd.read_csv('datasets/sign-language-mnist/sign_mnist_test.csv')


im2 = im1.reshape(1,28,28,1)
#Storing Pixel array in form length width and channel in df_x
test_data = test.iloc[:1,1:].values.reshape(1,28,28,1)
print im2

#Storing the labels in y
#test_labe = test.iloc[:,0].values
print type(test_data),type(im1)
#test_data = np.array(test_data)

model1 = load_model('my_model.h5')

model1.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

#model = load_model('my_model.h5')
#print test_data
'''
im1 = im1.tolist()
print im1
for i in range(len(im1)):
    for j in range(len(im1[i])):
        n = im1[i][j]
        n1 = []
        n1 = [n] 
'''
 #       im1[i][j].append(n1)
#print im1
prict = model1.predict(im2)
#print prict
#prict = np.argmax(np.round(prict),axis=1)
#print prict.shape,test_label.shape
sol = np.argmax(prict,axis=1)
#print test_data
#print im1
m = sol[0]
print sign[m]

                                                                                                                                                                                  
                           
