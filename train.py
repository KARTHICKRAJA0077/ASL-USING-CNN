import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix

test = pd.read_csv('datasets/sign-language-mnist/sign_mnist_test.csv')

#Storing Pixel array in form length width and channel in df_x
test_data = test.iloc[:,1:].values.reshape(len(test),28,28,1)

#Storing the labels in y
test_labe = test.iloc[:,0].values

test_data = np.array(test_data)

model1 = load_model('my_model.h5')

model1.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

#model = load_model('my_model.h5')

prict = model1.predict(test_data)
#print prict
#prict = np.argmax(np.round(prict),axis=1)
#print prict.shape,test_label.shape
print np.argmax(prict,axis=1)
print test_labe
con_mat = confusion_matrix(test_labe, np.argmax(prict,axis=1))
c=0
for i in con_mat:
    for j in range(6):
	    print i[j],
    print ""
    if c>4:
        break 
    c+=1
	 
	
	
