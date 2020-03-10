import tensorflow as tf
import keras
from keras.layers import Dense,Dropout,Activation,Flatten
import sys,os
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
import numpy as np
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
import pandas as pd

data=pd.read_csv('icml_face_data.csv')
data.to_numpy()
# data.head() returns the first 5 rows of your dataframe 
#data.tail() returns the last 5 rows of your dataframe
x_train,y_train,x_test,y_test=[],[],[],[]
#iterrows() helps you iterate over rows of the dataframe
for index,row in data.iterrows():
    val=row[' pixels'].split(" ") #2space and then pixel hai is column ka name
    try:
        if 'Training' in row[' Usage']:
            x_train.append(np.array(val,'float32')) 
            y_train.append(row['emotion'])
        elif 'PublicTest' in row[' Usage']:
            x_test.append(np.array(val,'float32'))
            y_test.append(row['emotion'])
    except:
        print(f'Error occured at index:{index} and row:{row}')

x_train=np.array(x_train,'float32')
x_test=np.array(x_test,'float32')
y_train=np.array(y_train,'float32')
y_test=np.array(y_test,'float32')

#Normalising the data : bringing values to  [0,1) 

x_train-=np.mean(x_train,axis=0)
x_train/=np.std(x_train,axis=0)


x_test-=np.mean(x_test,axis=0)
x_test/=np.std(x_test,axis=0)

n_features=64
n_labels=7
batch_size=64
epochs=30
width,height=48,48
x_train=x_train.reshape(x_train.shape[0],width,height,1)
#1 here means : no of features that is 1
x_test=x_test.reshape(x_test.shape[0],width,height,1)

y_train=np_utils.to_categorical(y_train,num_classes=n_labels)
y_test=np_utils.to_categorical(y_test,num_classes=n_labels)

from keras.models import Sequential 

#Designing CNN 
model=Sequential()
#first layer
model.add(Conv2D(n_features,(3,3),input_shape=(x_train.shape[1:]),activation='relu'))
model.add(Conv2D(n_features,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#second layer
model.add(Conv2D(n_features,(3,3),activation='relu'))
model.add(Conv2D(n_features,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


#third layer
model.add(Conv2D(n_features,(3,3),activation='relu'))
model.add(Conv2D(n_features,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(16*n_features,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(16*n_features,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(n_labels,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test),shuffle=True)

#saving the weights of our model
fer_json = model.to_json()
with open('fer.json', "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
print('Saved Model to disk..!')