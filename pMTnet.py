import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import csv
import random
import os
from io import StringIO
from collections import Counter
import keras
from keras.layers import Input,Dense,concatenate,Dropout
from keras.models import Model,load_model                                                      
from keras import backend as K
##Customer Input
#python pMTnet.py -input input.csv -library library_dir -output output_dir
args = sys.argv
file_dir=args[args.index('-input')+1] #input protein seq file
library_dir=args[args.index('-library')+1] #directory to downloaded library

model_dir=library_dir+'/h5_file'
aa_dict_dir=library_dir+'/Atchley_factors.csv' #embedding vector for tcr encoding
hla_db_dir=library_dir+'/hla_library/' #hla sequence
output_dir=args[args.index('-output')+1] #diretory to hold encoding and prediction output
output_log_dir=args[args.index('-output_log')+1] #standard output
################################
# Reading Encoding Matrix #
################################
########################### Atchley's factors#######################
aa_dict_atchley=dict()
with open(aa_dict_dir,'r') as aa:
    aa_reader=csv.reader(aa)
    next(aa_reader, None)
    for rows in aa_reader:
        aa_name=rows[0]
        aa_factor=rows[1:len(rows)]
        aa_dict_atchley[aa_name]=np.asarray(aa_factor,dtype='float')
########################### One Hot ##########################   
aa_dict_one_hot = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,
           'M': 10,'N': 11,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,
           'W': 18,'Y': 19,'X': 20}  # 'X' is a padding variable
########################### Blosum ########################## 
BLOSUM50_MATRIX = pd.read_table(StringIO(u"""                                                                                      
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *                                                           
A  5 -2 -1 -2 -1 -1 -1  0 -2 -1 -2 -1 -1 -3 -1  1  0 -3 -2  0 -2 -2 -1 -1 -5                                                           
R -2  7 -1 -2 -4  1  0 -3  0 -4 -3  3 -2 -3 -3 -1 -1 -3 -1 -3 -1 -3  0 -1 -5                                                           
N -1 -1  7  2 -2  0  0  0  1 -3 -4  0 -2 -4 -2  1  0 -4 -2 -3  5 -4  0 -1 -5                                                           
D -2 -2  2  8 -4  0  2 -1 -1 -4 -4 -1 -4 -5 -1  0 -1 -5 -3 -4  6 -4  1 -1 -5                                                           
C -1 -4 -2 -4 13 -3 -3 -3 -3 -2 -2 -3 -2 -2 -4 -1 -1 -5 -3 -1 -3 -2 -3 -1 -5                                                           
Q -1  1  0  0 -3  7  2 -2  1 -3 -2  2  0 -4 -1  0 -1 -1 -1 -3  0 -3  4 -1 -5                                                           
E -1  0  0  2 -3  2  6 -3  0 -4 -3  1 -2 -3 -1 -1 -1 -3 -2 -3  1 -3  5 -1 -5                                                           
G  0 -3  0 -1 -3 -2 -3  8 -2 -4 -4 -2 -3 -4 -2  0 -2 -3 -3 -4 -1 -4 -2 -1 -5                                                           
H -2  0  1 -1 -3  1  0 -2 10 -4 -3  0 -1 -1 -2 -1 -2 -3  2 -4  0 -3  0 -1 -5                                                          
I -1 -4 -3 -4 -2 -3 -4 -4 -4  5  2 -3  2  0 -3 -3 -1 -3 -1  4 -4  4 -3 -1 -5                                                           
L -2 -3 -4 -4 -2 -2 -3 -4 -3  2  5 -3  3  1 -4 -3 -1 -2 -1  1 -4  4 -3 -1 -5                                                           
K -1  3  0 -1 -3  2  1 -2  0 -3 -3  6 -2 -4 -1  0 -1 -3 -2 -3  0 -3  1 -1 -5                                                           
M -1 -2 -2 -4 -2  0 -2 -3 -1  2  3 -2  7  0 -3 -2 -1 -1  0  1 -3  2 -1 -1 -5                                                           
F -3 -3 -4 -5 -2 -4 -3 -4 -1  0  1 -4  0  8 -4 -3 -2  1  4 -1 -4  1 -4 -1 -5                                                           
P -1 -3 -2 -1 -4 -1 -1 -2 -2 -3 -4 -1 -3 -4 10 -1 -1 -4 -3 -3 -2 -3 -1 -1 -5                                                           
S  1 -1  1  0 -1  0 -1  0 -1 -3 -3  0 -2 -3 -1  5  2 -4 -2 -2  0 -3  0 -1 -5                                                           
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  2  5 -3 -2  0  0 -1 -1 -1 -5                                                           
W -3 -3 -4 -5 -5 -1 -3 -3 -3 -3 -2 -3 -1  1 -4 -4 -3 15  2 -3 -5 -2 -2 -1 -5                                                           
Y -2 -1 -2 -3 -3 -1 -2 -3  2 -1 -1 -2  0  4 -3 -2 -2  2  8 -1 -3 -1 -2 -1 -5                                                           
V  0 -3 -3 -4 -1 -3 -3 -4 -4  4  1 -3  1 -1 -3 -2  0 -3 -1  5 -3  2 -3 -1 -5                                                           
B -2 -1  5  6 -3  0  1 -1  0 -4 -4  0 -3 -4 -2  0  0 -5 -3 -3  6 -4  1 -1 -5                                                           
J -2 -3 -4 -4 -2 -3 -3 -4 -3  4  4 -3  2  1 -3 -3 -1 -2 -1  2 -4  4 -3 -1 -5                                                           
Z -1  0  0  1 -3  4  5 -2  0 -3 -3  1 -1 -4 -1  0 -1 -2 -2 -3  1 -3  5 -1 -5                                                           
X -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -5                                                           
* -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5  1                                                           
"""), sep='\s+').loc[list(aa_dict_one_hot.keys()), list(aa_dict_one_hot.keys())]
assert (BLOSUM50_MATRIX == BLOSUM50_MATRIX.T).all().all()

ENCODING_DATA_FRAMES = {
    "BLOSUM50": BLOSUM50_MATRIX,
    "one-hot": pd.DataFrame([
        [1 if i == j else 0 for i in range(len(aa_dict_one_hot.keys()))]
        for j in range(len(aa_dict_one_hot.keys()))
    ], index=aa_dict_one_hot.keys(), columns=aa_dict_one_hot.keys())
}

########################### HLA pseudo-sequence ##########################
#pMHCpan 
HLA_ABC=[hla_db_dir+'/A_prot.fasta',hla_db_dir+'/B_prot.fasta',hla_db_dir+'/C_prot.fasta',hla_db_dir+'/E_prot.fasta']
HLA_seq_lib={}
for one_class in HLA_ABC:
    prot=open(one_class)
    #pseudo_seq from netMHCpan:https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000796
    pseudo_seq_pos=[7,9,24,45,59,62,63,66,67,79,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,152,156,158,159,163,167,171]
    #write HLA sequences into a library
    #class I alles
    name=''
    sequence=''                                                                                                                        
    for line in prot:
        if len(name)!=0:
            if line.startswith('>HLA'):
                pseudo=''
                for i in range(0,33):
                    if len(sequence)>pseudo_seq_pos[i]:
                        pseudo=pseudo+sequence[pseudo_seq_pos[i]]
                HLA_seq_lib[name]=pseudo
                name=line.split(' ')[1]
                sequence=''
            else:
                sequence=sequence+line.strip()
        else:
            name=line.split(' ')[1]
########################################
# Input data encoding helper functions #
########################################
#################functions for TCR encoding####################
def preprocess(filedir):
    #Preprocess TCR files                                                                                                                 
    print('Processing: '+filedir)
    if not os.path.exists(filedir):
        print('Invalid file path: ' + filedir)
        return 0
    dataset = pd.read_csv(filedir, header=0)
    dataset = dataset.sort_values('CDR3').reset_index(drop=True)
    #Preprocess HLA_antigen files
    #remove HLA which is not in HLA_seq_lib; if the input hla allele is not in HLA_seq_lib; then the first HLA startswith the input HLA allele will be given     
    #Remove antigen that is longer than 15aa
    dataset=dataset.dropna()
    HLA_list=set(dataset['HLA'])
    HLA_to_drop = list()
    for i in HLA_list:
        if len([hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(i))])==0:
            HLA_to_drop.append(i)
            print('drop '+i)
    dataset=dataset[~dataset['HLA'].isin(HLA_to_drop)]
    dataset=dataset[dataset.Antigen.str.len()<16]
    print(str(max(dataset.index)-dataset.shape[0]+1)+' antigens longer than 15aa are dropped!')
    TCR_list=dataset['CDR3'].tolist()
    antigen_list=dataset['Antigen'].tolist()
    HLA_list=dataset['HLA'].tolist()
    return TCR_list,antigen_list,HLA_list

def aamapping_TCR(peptideSeq,aa_dict):
    #Transform aa seqs to Atchley's factors.                                                                                              
    peptideArray = []
    if len(peptideSeq)>80:
        print('Length: '+str(len(peptideSeq))+' over bound!')
        peptideSeq=peptideSeq[0:80]
    for aa_single in peptideSeq:
        try:
            peptideArray.append(aa_dict[aa_single])
        except KeyError:
            print('Not proper aaSeqs: '+peptideSeq)
            peptideArray.append(np.zeros(5,dtype='float32'))
    for i in range(0,80-len(peptideSeq)):
        peptideArray.append(np.zeros(5,dtype='float32'))
    return np.asarray(peptideArray)

def hla_encode(HLA_name,encoding_method):
    #Convert the a HLA allele to a zero-padded numeric representation.
    if HLA_name not in HLA_seq_lib.keys():
        HLA_name=[hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(HLA_name))][0]
    if HLA_name not in HLA_seq_lib.keys():
        print('Not proper HLA allele:'+HLA_name)
    HLA_sequence=HLA_seq_lib[HLA_name]
    HLA_int=[aa_dict_one_hot[char] for char in HLA_sequence]
    if len(HLA_int)!=34:
        k=len(HLA_int)
        HLA_int.extend([20] * (34 - k))
    result=ENCODING_DATA_FRAMES[encoding_method].iloc[HLA_int]
    # Get a numpy array of 34 rows and 21 columns
    return np.asarray(result)

def peptide_encode_HLA(peptide, maxlen,encoding_method):
    #Convert peptide amino acid sequence to numeric encoding
    if len(peptide) > maxlen:
        msg = 'Peptide %s has length %d > maxlen = %d.'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    peptide= peptide.replace(u'\xa0', u'').upper()    #remove non-breaking space  
    o = [aa_dict_one_hot[aa] if aa in aa_dict_one_hot.keys() else 20 for aa in peptide] 
    #if the amino acid is not valid, replace it with padding aa 'X':20
    k = len(o)
    #use 'X'(20) for padding
    o = o[:k // 2] + [20] * (int(maxlen) - k) + o[k // 2:]
    if len(o) != maxlen:
        msg = 'Peptide %s has length %d < maxlen = %d, but pad is "none".'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    result=ENCODING_DATA_FRAMES[encoding_method].iloc[o]
    return np.asarray(result)

def TCRMap(dataset,aa_dict):
    #Wrapper of aamapping                                                                                                                 
    pos = 0
    TCR_counter = Counter(dataset)
    TCR_array = np.zeros((len(dataset), 80, 5, 1), dtype=np.float32)
    for sequence, length in TCR_counter.items():
        TCR_array[pos:pos+length] = np.repeat(aamapping_TCR(sequence,aa_dict).reshape(1,80,5,1), length, axis=0)
        pos += length
    print('TCRMap done!')
    return TCR_array

def HLAMap(dataset,encoding_method):
    #Input a list of HLA and get a three dimentional array
    pos=0
    HLA_array = np.zeros((len(dataset), 34, 21), dtype=np.int8)
    HLA_seen = dict()
    for HLA in dataset:
        if HLA not in HLA_seen.keys():  
            HLA_array[pos] = hla_encode(HLA,encoding_method).reshape(1,34,21)
            HLA_seen[HLA] = HLA_array[pos]
        else:
            HLA_array[pos] = HLA_seen[HLA]
        pos += 1
    print('HLAMap done!')
    return HLA_array

def antigenMap(dataset,maxlen,encoding_method):
    #Input a list of antigens and get a three dimentional array
    pos=0
    antigen_array = np.zeros((len(dataset), maxlen, 21), dtype=np.int8)
    antigens_seen = dict()
    for antigen in dataset:
        if antigen not in antigens_seen.keys():
            antigen_array[pos]=peptide_encode_HLA(antigen, maxlen,encoding_method).reshape(1,maxlen,21)
            antigens_seen[antigen] = antigen_array[pos]
        else:
            antigen_array[pos] = antigens_seen[antigen]
        pos += 1
    print('antigenMap done!')
    return antigen_array

def pearson_correlation_f(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred) #being K.mean a scalar here, it will be automatically subtracted from all elements in y_pred                
    fst = y_true - K.mean(y_true)
    devP = K.std(y_pred)
    devT = K.std(y_true)
    return K.mean(fsp*fst)/(devP*devT)

def pos_neg_acc(y_true,y_pred):
    #self-defined prediction accuracy metric
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.cast(negative_pred<positive_pred,"float16"))
    return diff

def pos_neg_loss(y_true,y_pred):
    #self-defined prediction loss function 
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.relu(1+negative_pred-positive_pred))+0.2*K.mean(K.square(negative_pred)+K.square(positive_pred))
    return diff

#########################################                                                                                                      
# preprocess input data and do encoding #                                                                                                      
#########################################
#Read data
#TCR Data preprocess                                                                                                                      
log_file=open(output_log_dir,'w')
sys.stdout=log_file
print('Mission loading.')

TCR_list,antigen_list,HLA_list=preprocess(file_dir)
TCR_array=TCRMap(TCR_list,aa_dict_atchley)
antigen_array=antigenMap(antigen_list,15,'BLOSUM50')
HLA_array=HLAMap(HLA_list,'BLOSUM50')

#Model prediction                                                                                                                         
TCR_encoder=load_model(model_dir+'/TCR_encoder_30.h5')
TCR_encoder=Model(TCR_encoder.input,TCR_encoder.layers[-12].output)
TCR_encoded_result=TCR_encoder.predict(TCR_array)

HLA_antigen_encoder=load_model(model_dir+'/HLA_antigen_encoder_60.h5',custom_objects={'pearson_correlation_f': pearson_correlation_f})
HLA_antigen_encoder=Model(HLA_antigen_encoder.input,HLA_antigen_encoder.layers[-2].output)
HLA_antigen_encoded_result=HLA_antigen_encoder.predict([antigen_array,HLA_array])

TCR_encoded_matrix=pd.DataFrame(data=TCR_encoded_result,index=range(1,len(TCR_list)+1))
HLA_antigen_encoded_matrix=pd.DataFrame(data=HLA_antigen_encoded_result,index=range(1,len(HLA_list)+1))
# TCR_encoded_matrix.to_csv(output_dir+'/TCR_output.csv',sep=',')
# HLA_antigen_encoded_matrix.to_csv(output_dir+'/MHC_antigen_output.csv',sep=',')
print('Encoding Accomplished.\n')
#########################################                                                                                                                       
# make prediction based on encoding     #                                                                                                                     
#########################################   
############## Load Prediction Model ################                                                                                                           
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
ternary_prediction.load_weights(model_dir+'/weights.h5')
################ read dataset #################                                                                                                                  
#read background negative TCRs
# This way we don't need to save and reload the matrices and use double de memory.
TCR_neg_df_1k=pd.read_csv(library_dir+'/bg_tcr_library/TCR_output_1k.csv', names=pd.RangeIndex(0, 30,1), header=None, skiprows=1) 
TCR_neg_df_10k=pd.read_csv(library_dir+'/bg_tcr_library/TCR_output_10k.csv', names=pd.RangeIndex(0, 30,1), header=None, skiprows=1)
# As of the state of the software this step looks redundant and a waste of memory as it is loading an object that is already in memory but using a new variable name
# TCR_pos_df=pd.read_csv(output_dir+'/TCR_output.csv',index_col=0)  
# MHC_antigen_df=pd.read_csv(output_dir+'/MHC_antigen_output.csv',index_col=0)
################ make prediction ################# 
rank_output=[]
for each_data_index in range(TCR_encoded_matrix.shape[0]):
    tcr_pos=TCR_encoded_matrix.iloc[[each_data_index,]]
    pmhc=HLA_antigen_encoded_matrix.iloc[[each_data_index,]]
    #used the positive pair with 1k negative tcr to form a 1001 data frame for prediction                                                                      

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

rank_output_matrix=pd.DataFrame({'CDR3':TCR_list,'Antigen':antigen_list,'HLA':HLA_list,'Rank':rank_output},index=range(1,len(TCR_list)+1))
rank_output_matrix.to_csv(output_dir+'/prediction.csv',sep=',', index=False)
print('Prediction Accomplished.\n')
log_file.close()
#delete encoding files
#os.remove(output_dir+'/MHC_antigen_output.csv')
#os.remove(output_dir+'/TCR_output.csv')
