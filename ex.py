'''
test= np.genfromtxt('/Users/diwakar/Downloads/sign-language-mnist/sign_mnist_test.csv',delimiter=",")
train= np.genfromtxt('/Users/diwakar/Downloads/sign-language-mnist/sign_mnist_train.csv',delimiter=",")
'''
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import load_model
train= pd.read_csv('datasets/sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('datasets/sign-language-mnist/sign_mnist_test.csv')
#print data.head()
#Storing Pixel array in form length width and channel in df_x
train_data = train.iloc[:,1:].values.reshape(len(train),28,28,1)

#Storing the labels in y
train_labe = train.iloc[:,0].values

#Storing Pixel array in form length width and channel in df_x
test_data = test.iloc[:,1:].values.reshape(len(test),28,28,1)

#Storing the labels in y
test_labe = test.iloc[:,0].values

train_label = keras.utils.to_categorical(train_labe,num_classes=26)
test_label = keras.utils.to_categorical(test_labe,num_classes=26)

train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)
'''
print train_data.shape
print train_label.shape
print test_data.shape
print test_label.shape
'''
#print train_data.shape,train_label.shape
#CNN model
model = Sequential()
model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(180))
model.add(Dropout(0.6))
model.add(Dense(26))
model.add(Activation('softmax'))
#model.add(Activation('softmax'))
#model.add(Activation('softmax'))
#model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

#print model.summary()

#fitting it with just 100 images for testing 

model.fit(train_data,train_label,epochs=10,validation_data=(test_data,test_label))
print model.evaluate(test_data,test_label)

model.save('my_model1.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model1.h5')

prict = model.predict(test_data[0:100])
#print prict
#prict = np.argmax(np.round(prict),axis=1)
print prict.shape,test_label.shape
print np.argmax(prict,axis=1)
