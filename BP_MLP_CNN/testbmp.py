import matplotlib.pyplot as plt  
from PIL import Image

from mlp import Model
from mlp import BasicRoutinizer
from preprocess import Data
import numpy as np
from tqdm import tqdm
import os
from matplotlib import rc

source_folder = input("Where is the test set?")
input_matrix = []
label_vector = []

for root, dirs, files in tqdm(os.walk(source_folder)):
    for file in files:
        file_path = os.path.join(root, file)
        digit = int(root.lstrip(source_folder + '\\'))
        I = Image.open(file_path,mode="r")  
        a = np.asarray(I, dtype=int)
        a = a.reshape(-1,1)
        input_matrix.append(a)
        label_vector.append(digit-1)

input_matrix = np.array(input_matrix)
label_vector = np.array(label_vector).reshape(1,-1)
input_matrix = input_matrix.squeeze(2).T

m = Model(layer_size=[784,500,12],debug=False,initial_scalar=0.01,dropout=0.5)
m.resume_from_ckpt(o_path="./ckpt")

meanstd = np.load("meanstd.npz")
mean = meanstd["mean"]
print(mean.shape)
std = meanstd["std"]
print(std.shape)
input_matrix = (input_matrix - mean) / std

predict = m.forward(input_matrix,train=False)

npredict = m.logits2res(predict)

acc = m.acc(real_data=label_vector,predict_data=npredict)
print(acc)