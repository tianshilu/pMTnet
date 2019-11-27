import random
import pandas as pd
import csv
import sys
import keras
import numpy as np
from keras.layers import Dense,Input,concatenate,Dropout,BatchNormalization
from keras.optimizers import Adam, Adamax, RMSprop, Adagrad, Adadelta, Nadam
from keras import backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model,load_model  
random.seed(54321)
############# Customer Input ###################
args = sys.argv
model_dir=args[args.index('-model')+1]
encoding=args[args.index('-encoding_folder')+1]            
output_dir=args[args.index('-output_dir')+1]

############# Define Functions ################ 
def pos_neg_acc(y_true,y_pred):
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.cast(negative_pred<positive_pred,"float16"))
    return diff

def pos_neg_loss(y_true,y_pred):
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.relu(1+negative_pred-positive_pred))+0.2*K.mean(K.square(negative_pred)+K.square(positive_pred))
    return diff

############## Load Network ################
#set up model
hla_antigen_in=Input(shape=(60,),name='hla_antigen_in')
pos_in=Input(shape=(30,),name='pos_in')
ternary_layer1_pos=concatenate([pos_in,hla_antigen_in])
ternary_dense1=Dense(300,activation='relu')(ternary_layer1_pos)
ternary_do1=Dropout(0.2)(ternary_dense1)
ternary_dense2=Dense(200,activation='relu')(ternary_do1)
ternary_dense3=Dense(100,activation='relu')(ternary_dense2)
ternary_output=Dense(1,activation='linear')(ternary_dense3)
ternary_prediction=Model(inputs=[pos_in,hla_antigen_in],outputs=ternary_output)
#load weights
ternary_prediction.load_weights(model_dir)

################ read dataset #################
TCR_neg_df_1k=pd.read_csv('/project/bioinformatics/Xiao_lab/s171162/HLA/ternary/data/negative_TCR/encoding/TCR_output.csv',index_col=0)
TCR_neg_df_10k=pd.read_csv('/project/bioinformatics/Xiao_lab/s171162/HLA/ternary/data/negative_TCR/encoding_10k/TCR_output.csv',index_col=0)
TCR_pos_df=pd.read_csv(encoding+'/TCR_output.csv',index_col=0)
MHC_antigen_df=pd.read_csv(encoding+'/MHC_antigen_output.csv',index_col=0)
print(TCR_pos_df.shape)
rank_output=[]
for each_data_index in range(TCR_pos_df.shape[0]):
    print(each_data_index)
    tcr_pos=TCR_pos_df.iloc[[each_data_index,]]
    pmhc=MHC_antigen_df.iloc[[each_data_index,]]
    #used the positive pair with 10000 negative tcr to form a 1001 data frame for prediction    
    TCR_input_df=pd.concat([tcr_pos,TCR_neg_df_1k],axis=0)
    MHC_antigen_input_df= pd.DataFrame(np.repeat(pmhc.values,1001,axis=0))
    prediction=ternary_prediction.predict({'pos_in':TCR_input_df,'hla_antigen_in':MHC_antigen_input_df})
    
    rank=1-(sorted(prediction.tolist()).index(prediction.tolist()[0])+1)/1000
    #if rank is higher than top 2% use 10k background TCR
    if rank<0.02:
        TCR_input_df=pd.concat([tcr_pos,TCR_neg_df_10k],axis=0)
        MHC_antigen_input_df= pd.DataFrame(np.repeat(pmhc.values,10001,axis=0))
        prediction=ternary_prediction.predict({'pos_in':TCR_input_df,'hla_antigen_in':MHC_antigen_input_df})

        rank=1-(sorted(prediction.tolist()).index(prediction.tolist()[0])+1)/10000
    rank_output.append(rank)


with open(output_dir+'_prediction.csv', 'a') as csv_file:
    csv.writer(csv_file).writerow(rank_output)
csv_file.close()
