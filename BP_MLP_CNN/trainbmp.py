import matplotlib.pyplot as plt  
from PIL import Image

from mlp import Model
from mlp import BasicRoutinizer
from preprocess import Data
import numpy as np
from tqdm import tqdm
import os
from matplotlib import rc

source_folder = './trainnew'
val_source_folder='./validationnew'
input_matrix = []
label_vector = []
val_input_matrix = []
val_label_vector = []

for root, dirs, files in tqdm(os.walk(source_folder)):
    for file in files:
        file_path = os.path.join(root, file)
        digit = int(root.lstrip(source_folder + '\\'))
        # print(f'-{digit}-{file}') 
        # print(f'-{file_path}')
        #  
        I = Image.open(file_path,mode="r")  
        a = np.asarray(I, dtype=int)
        a = a.reshape(-1,1)
        input_matrix.append(a)
        label_vector.append(digit-1)

for root, dirs, files in tqdm(os.walk(val_source_folder)):
    for file in files:
        file_path = os.path.join(root, file)
        digit = int(root.lstrip(val_source_folder + '\\'))
        # print(f'-{digit}-{file}') 
        # print(f'-{file_path}')
        #  
        I = Image.open(file_path,mode="r")  
        a = np.asarray(I, dtype=int)
        a = a.reshape(-1,1)
        val_input_matrix.append(a)
        val_label_vector.append(digit-1)

input_matrix = np.array(input_matrix)
label_vector = np.array(label_vector).reshape(1,-1)
input_matrix = input_matrix.squeeze(2).T

val_input_matrix = np.array(val_input_matrix)
val_label_vector = np.array(val_label_vector).reshape(1,-1)
val_input_matrix = val_input_matrix.squeeze(2).T

# print(label_vector.shape)
# print(input_matrix.shape)

d = Data(y_train=label_vector,X_train=input_matrix,y_val=val_label_vector,X_val=val_input_matrix)
d.shuffle()
d.normalize()


# Perhaps pre_process is needed

# Cannot create val_set in this way
# The distribution of data is not random



max_iter = 10

m = Model(layer_size=[784,500,12],debug=False,initial_scalar=0.01,dropout=0.5)
m.initialize()
#m.resume_from_ckpt()
#m.resume_from_ckpt(o_path="./ckpt")
run = BasicRoutinizer(max_iter=max_iter,feature_data=d.X_train,label_data=d.y_train,lr=3e-4,dynamic_show=True,dynamic_show_res=1,type='CrossEntropy',val_feature_data=d.X_val,val_label_data=d.y_val,min_loss=2.5,batchsize=1,acc_b=True,max_acc=0.8)
run.run(m=m)


# 100 100 1e-5 0.01 continues to decrease slow at first
# 784 500 12, 0.01 lr=1e-5 accuracy is good
# try deeper model

#[784,64,16,12]  [784,32,32,32,32,12]