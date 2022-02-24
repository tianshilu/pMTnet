import random
import pandas as pd
import csv
import sys
import keras
import numpy as np
from keras.layers import Dense,Input,concatenate,Dropout,BatchNormalization,LSTM,Reshape
from keras.models import Model
from keras.optimizers import Adam, Adamax, RMSprop, Adagrad, Adadelta, Nadam
from keras import backend as K
from keras import regularizers
#random.seed(54321)
############# Define function  ################ 

def pos_neg_acc(y_true,y_pred):
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.cast(negative_pred<positive_pred,"float16"))
    return diff

def pos_neg_loss(y_true,y_pred):
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.relu(1+negative_pred-positive_pred))+0.03*K.mean(K.square(negative_pred)+K.square(positive_pred))
    return diff

############# Read training and testing data  ################

#train
tcr_file_train_pos='positive/TCR_output.csv'
tcr_file_train_neg='negative/TCR_output.csv'                                        
hla_antigen_file_train='MHC_antigen_output.csv'
############# Read encoding ################
# read encoders
tcr_train_pos=pd.read_csv(tcr_file_train_pos,index_col=0)
tcr_train_neg=pd.read_csv(tcr_file_train_neg,index_col=0)
hla_antigen_train=pd.read_csv(hla_antigen_file_train,index_col=0)

#dummy Y_train                                                                                                                    
Y_train = np.random.randint(2, size=(2,tcr_train_pos.shape[0])).T

##############network################

# 2 inputs:positive TCR and negative TCR
hla_antigen_in=Input(shape=(60,),name='hla_antigen_in')
pos_in=Input(shape=(30,),name='pos_in')
neg_in=Input(shape=(30,),name='neg_in')
    
ternary_layer1_pos=concatenate([pos_in,hla_antigen_in])
ternary_layer1_neg=concatenate([neg_in,hla_antigen_in])

ternary_dense1=Dense(300,activation='relu')
ternary_layer2_pos=ternary_dense1(ternary_layer1_pos)
ternary_layer2_neg=ternary_dense1(ternary_layer1_neg)

ternary_do1=Dropout(0.2)
ternary_layer3_pos=ternary_do1(ternary_layer2_pos)
ternary_layer3_neg=ternary_do1(ternary_layer2_neg)

ternary_dense2=Dense(200,activation='relu')
ternary_layer4_pos=ternary_dense2(ternary_layer3_pos)
ternary_layer4_neg=ternary_dense2(ternary_layer3_neg)

ternary_dense3=Dense(100,activation='relu')
ternary_layer6_pos=ternary_dense3(ternary_layer4_pos)
ternary_layer6_neg=ternary_dense3(ternary_layer4_neg)

ternary_output=Dense(1,activation='linear')
pos_out=ternary_output(ternary_layer6_pos)
neg_out=ternary_output(ternary_layer6_neg)
    
merged_vector=concatenate([neg_out,pos_out],axis=-1,name='output')

ternary_prediction=Model(inputs=[pos_in,neg_in,hla_antigen_in],outputs=merged_vector)
ternary_prediction.compile(optimizer=Adamax(),loss=[pos_neg_loss],metrics=[pos_neg_acc])
ternary_prediction.summary()
ternary_prediction.fit({'pos_in':tcr_train_pos,'neg_in':tcr_train_neg,'hla_antigen_in':hla_antigen_train},{'output':Y_train},epochs=150,batch_size=256,shuffle=True)
                                

