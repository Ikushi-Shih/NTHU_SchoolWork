# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:44:30 2017

@author: Darcy
"""
import numpy as np
import h5py as h5py
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

#Initialize the CNN
classifier = Sequential()
#step1
#classifier.add(Convolution2D(32,3,3,input_shape= (64,64,3), activation = 'relu'))
#
##step2
#classifier.add(MaxPooling2D(pool_size = (2,2)))
#
##step1
#classifier.add(Convolution2D(64,3,3, activation = 'relu'))
#
##step2
#classifier.add(MaxPooling2D(pool_size = (2,2)))
#
##step1
#classifier.add(Convolution2D(128,3,3, activation = 'relu'))
#
##step2
#classifier.add(MaxPooling2D(pool_size = (2,2)))
#step3
classifier.add(Dense(output_dim = 128, activation = 'relu',input_shape= (64,64,3)))
classifier.add(Flatten())


#step4

#classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim = 64, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# import OS
import os 

#setwd
os.chdir('C:/Users/xd944/Documents/20171212/original/ANN-result/')

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/xd944/Documents/20171212/original/training/',
        target_size=(64,64),
        batch_size=12,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/xd944/Documents/20171212/original/testing/',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
a = 1
for a in range(1,6):
    history = classifier.fit_generator(training_set,
            steps_per_epoch=24,
            epochs=50,
            validation_data=test_set,
            validation_steps=6)
    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("ANN_Accuracy_original_"+str(a)+".png")
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("ANN_Loss_original_"+str(a)+".png")
    
    classifier.save('20171212_ANN_original_'+str(a)+'.h5')
    
    
    training_set.class_indices
    datapath = 'C:/Users/xd944/Documents/20171212/original/Predict/'
    #classifier = load_model('20171212_CNN_original.h5')
    
    
    from keras.preprocessing import image
    img_list = os.listdir(datapath)
        
    results = []
    for i in range(0,len(img_list),1):
        test_image = image.load_img(datapath + img_list[i],target_size=(64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis = 0)
        results.append(classifier.predict(test_image))
        if results[i][0] == 0:
            prediction = 'Glaucoma'
        else:
            prediciton = 'Healthy'
        
    Final = np.hstack(results)
    Final = Final.T
    
    name_list = img_list
    i=0
    for i in range(0,len(img_list)):
        name_list[i] = img_list[i][0:9] #+ new[i][-8:]
        i=i+1
        
    Final_list = np.column_stack([name_list,Final])
    index = ['Row'+str(i) for i in range(1, len(Final_list)+1)]
    test = pd.DataFrame(data = Final_list[0:,0:],columns = ["ID","Prediction"],index=index)
    test = test.astype(dtype= {"ID":"object","Prediction":"float"})
    Final_list = pd.DataFrame(data = test.groupby(["ID"], as_index=False).mean())
    
    list_AB=[None]*len(Final_list)
    i=0
    for A in Final_list.iloc[0:,0]:
        list_AB[i] = A[0]
        i=i+1
            
    count_A = list_AB.count("A")
    count_B = list_AB.count("B")
    
    Final_list[Final_list["Prediction"] >= 0.5] = 1
    Final_list[Final_list["Prediction"] < 0.5] = 0
    
    H = count_A
    G = count_B
    Specificity = sum(Final_list.iloc[0:H,1])/H
    Sensitivity = 1 - sum(Final_list.iloc[H:len(Final_list),1])/G
    Accuracy = (Specificity*H + Sensitivity*G)/(H+G)
    print("Specificity: {}".format(Specificity))
    print("Sensitivity: {}".format(Sensitivity))
    print("Accuracy: {}".format(Accuracy))
    
    f = open("output_"+str(a)+'.txt',"w") #opens file with name of "test.txt"
    f.write("Sensitivity:"+ str(Sensitivity) +"\n")
    f.write("Specificity:"+ str(Specificity) + "\n")
    f.write("Accuracy:"+  str(Accuracy))
    f.close()
    
