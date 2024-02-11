import numpy as np
from pydantic import BaseModel
from typing import Optional,List,Any
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from matplotlib import rc

class Layer(BaseModel):

    d_in:int
    d_out:int
    random_seed:Optional[int] = 42
    initial_scalar:Optional[float] = 0.1

    bias:Optional[np.ndarray] = None
    weight:Optional[np.ndarray] = None
    in_tensor:Optional[np.ndarray] = None
    d_weight:Optional[np.ndarray] = None
    d_bias:Optional[np.ndarray] = None
    d_in_tensor:Optional[np.ndarray] = None

    out:Optional[np.ndarray] = None

    saved_weight:Optional[np.ndarray] = None
    saved_bias:Optional[np.ndarray] = None

    momentum1:Optional[Any] = 0
    momentum2:Optional[Any] = 0
    class Config:
        arbitrary_types_allowed = True

    def initialize(self):
        np.random.seed(self.random_seed)
        self.weight = self.initial_scalar * np.random.random([self.d_out,self.d_in])
        np.random.seed(self.random_seed)
        self.bias = self.initial_scalar * np.random.random([self.d_out,1])

    def forward(self,tensor:np.ndarray,train:bool=True):
        # Should store some matrix that may be needed in the back_propagate process
        # The input tensor should be shape like (Features,Nums)
        if train is True:
            self.in_tensor = tensor
        out = np.dot(self.weight,tensor) + self.bias
        if train is True:
            self.out = out
        return out
    
    def back_propagate(self,upstream_derivative:np.ndarray):
        # The idea is: at every node, the current derivative is:
        # We pass on the dL/dx derivative
        
        # f = W(m,n)x(n,N) + b(m,1)one(1,N)
        # dL/dW = dL/df (m,N) * x.T (N,n)
        # dL/dx = W.T (n,m) * dL/df (m,N) 
        # dL/db =  dL/df (m,N) * one.T (N,1)

        # We include the back_propagate in the nonlinearity
        self.d_weight = np.dot(upstream_derivative, self.in_tensor.T)
        self.d_bias = np.dot(upstream_derivative,np.ones((np.size(upstream_derivative,1),1)))
        self.d_in_tensor = np.dot(self.weight.T, upstream_derivative)
        return self.d_in_tensor
    


class Loss(BaseModel):
    type:Optional[str] = 'MSE'    # 0.5(in-res)^2; in-res
    input:np.ndarray
    result:np.ndarray
    probas:np.ndarray = None
    class Config:
        arbitrary_types_allowed = True

    def forward(self):
        if self.type == 'MSE':
            # Here, self.result should be a (C,N) vector
            result = 0.5 * np.square(np.subtract(self.input, self.result)).mean()
            return result
            
        elif self.type == 'CrossEntropy':
            # Here, self.result should be a (1,N) vector, where each element is a class
            # We turn a one-hot matrix into a one-dimension vector
            
            # Deprecated; Numerically not stable

            # N = np.size(self.input,1)
            # print(f"THis is logits:{self.input}")
            # exp_logits = np.exp(self.input)
            # print(f"THis is probas:{np.sum(exp_logits)}")
            # sum_exp = np.sum(exp_logits,0)
            # probas = exp_logits/sum_exp
            # print(f"THis is probas:{np.sum(probas)}")
            # t = np.c_[self.result.T,range(N)]
            # result = None
            # real_probas = probas[t[:,0],t[:,1]]
            # cross_entropy = -np.log2(real_probas)
            # result = np.mean(cross_entropy)

            # self.probas = probas
            # print(f"THis is result:{result}")
            # return result

            # New way to compute it together:
            N = np.size(self.input,1)
            max_x = np.max(self.input,0)
            x = self.input - max_x
            
            exp_logits = np.exp(x)
            sum_exp = np.sum(exp_logits,0)
            self.probas = exp_logits/sum_exp
            
            modified_x = x - np.log(sum_exp)
            t = np.c_[self.result.T,range(N)]
            result = -np.mean(modified_x[t[:,0],t[:,1]])
            
            del max_x,x,exp_logits,sum_exp,modified_x,t
            return result



    def back_propagate(self):
        if self.type == 'MSE':
            result = np.subtract(self.input, self.result)
            return result
            
        elif self.type == 'CrossEntropy':
            # We use the cross-entropy softmax loss, although Model.final_ans is logit, not probas
            # Still, let us use one-hot matrix as it is very useful
            N = np.size(self.input,1)
            onehot = np.zeros_like(self.probas)
            t = np.c_[self.result.T,range(N)]
            onehot[t[:,0],t[:,1]] = 1
            result = self.probas - onehot

            del onehot,t
            return result
        pass

class DropoutLayer(BaseModel):
    dropout:float = 0.5
    mask:Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

    def forward(self,in_tensor:np.ndarray):
        # in_tensor.shape = (f,N)
        self.mask = np.random.uniform(0,1,in_tensor.shape)
        self.mask = np.where(self.mask>self.dropout,1/(1-self.dropout),0)
        out = in_tensor * self.mask
        return out

    def back_propagate(self,upstream_derivative:np.ndarray):
        derivative = upstream_derivative * self.mask  
        return derivative

class Nonlinearity(BaseModel):
    type:str = 'ReLU'
    in_tensor:Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

    def forward(self,tensor:np.ndarray,train:bool=True):
        if train is True:
            self.in_tensor = tensor
        if self.type == 'ReLU':
            tensor = np.maximum(tensor, 0)
            return tensor
        else:
            print(f"Don't support nonlinearity type:{self.type}\n")
        
    def back_propagate(self,upstream_derivative:np.ndarray,in_tensor:np.ndarray):
        # We only inplement the ReLU nonlinearity
        # f(x) = max(0,x)
        if self.type == 'ReLU':
            downstream_derivative = upstream_derivative * np.where(in_tensor >= 0,1,0)
        else:
            print(f"Don't support nonlinearity type:{self.type}\n")
        
        return downstream_derivative

class Model(BaseModel):
    """
    random_seed: An int setting the global random seed 
    layer_size:  A list representing the size of layers, including the last forward layer and the input size
    layer:       A list of class Layer
    debug:       A boolean showing whether debug info should appear
    """
    random_seed:Optional[int] = 42
    layer_size:list
    layer:Optional[List[Layer]] = []
    dropout_layer:Optional[List[DropoutLayer]] = []
    nonlinearity:Optional[Nonlinearity] = Nonlinearity(type='ReLU')
    debug:Optional[bool] = False 
    lr:Optional[float] = 1e-5
    dropout:Optional[float] = None

    initial_scalar:Optional[float] = 0.1

    final_ans:Optional[Any] = None
    loss:Optional[Loss] = None
    class Config:
        arbitrary_types_allowed = True


    def initialize(self):
        
        for i in range(len(self.layer_size)-1):
            layer = Layer(d_in=self.layer_size[i],d_out=self.layer_size[i+1],random_seed=self.random_seed,initial_scalar=self.initial_scalar)
            layer.initialize()
            self.layer.append(layer)
        
        if self.dropout is not None:
            for i in range(len(self.layer_size)-2):
                d = DropoutLayer(dropout=self.dropout)
                self.dropout_layer.append(d)
            

    def resume_from_ckpt(self,o_path:str="./ckpt"): 
        
        for i in range(len(self.layer_size)-1):
            path = o_path+f'/layer_{i}/ckpt.npz'
            ckpt = np.load(path)
            layer = Layer(d_in=self.layer_size[i],d_out=self.layer_size[i+1],random_seed=self.random_seed)
            layer.weight=ckpt["weight"]
            layer.bias=ckpt["bias"]
            self.layer.append(layer)

        if self.dropout is not None:
            for i in range(len(self.layer_size)-2):
                d = DropoutLayer(dropout=self.dropout)
                self.dropout_layer.append(d)
        



    def forward(self,tensor:np.ndarray,train:bool=True ):  
        if self.debug is True:
            print(f"A new iter of {len(self.layer)}:\n")
        for m in range(len(self.layer)):
            l =self.layer[m]
            if self.debug is True:
                print(f"[debug]Transposed tensor:{tensor.T}\n")
            
            tensor = l.forward(tensor=tensor,train=train)
            
            if self.debug is True:
                print(f"Layer{m}\n")
            if m<=len(self.layer)-2:
                if self.debug is True:
                    print(f"[debug]Waited Computed tensor:{tensor.T} at layer {m}\n")
                tensor = self.nonlinearity.forward(tensor=tensor,train=train)
            if self.debug is True:
                print(f"[debug]Computed tensor:{tensor.T}\n")
            if train is True:
                try:
                    tensor = self.dropout_layer[m].forward(tensor)
                except:
                    pass
            
        if train is False:
            
            return tensor
        
        self.final_ans = tensor
        return tensor
    
    def get_final_ans(self,type:str = 'softmax'):
        if type == 'softmax':
            exp_logits = np.exp(self.final_ans)
            sum_exp = np.sum(exp_logits,0)
            probas = exp_logits/sum_exp
            return probas
        
        elif type == 'direct':
            return self.final_ans
   
    def cal_loss(self,real_data:np.ndarray,type:str='MSE',train:bool=True,val_input=None):
        if train is True:
            self.loss = Loss(type=type,input=self.final_ans,result=real_data)
            return self.loss.forward()
        else:
            return Loss(type=type,input = val_input, result=real_data).forward()
    
    def logits2res(self,val_input=None):
        if val_input is None:
            logit = self.final_ans
        else:
            logit = val_input
        # logit [12,N]
        
        predict_label = np.argmax(logit,0).reshape(1,-1) 
        
        return predict_label

    def acc(self,real_data:np.ndarray,predict_data:np.ndarray):
        real = np.where(real_data-predict_data == 0,1,0)
        #print(predict_data)
        acc = np.sum(real) / np.size(real,1)
        return acc



    def back_propagate(self):
        # Set the loss function and compute
        # Set upstream_tensor to the last layer result.

        upstream_derivative = self.loss.back_propagate()

        for i in range(len(self.layer_size)-1):
            # The current layer
            current_layer = self.layer[len(self.layer_size)-2-i]
            if (self.debug is True):
                print(f"[debug]:{i} upstream derivative is {upstream_derivative}\n")
            if i > 0:
                if self.dropout is not None:
                    c_dropout_layer = self.dropout_layer[len(self.layer_size)-2-i]
                    upstream_derivative  = c_dropout_layer.back_propagate(upstream_derivative=upstream_derivative)
                upstream_derivative = self.nonlinearity.back_propagate(upstream_derivative=upstream_derivative,in_tensor=current_layer.out)
            if (self.debug is True):
                print(f"[debug]:{i} nonlinearity derivative: {upstream_derivative}\n")
            
            upstream_derivative=current_layer.back_propagate(upstream_derivative)

    def update(self,method:str='Adam',num_iter:int=0,beta1=0.9,beta2=0.999,debug=False):
        if method == 'SGD':
            for i in self.layer:
                i.weight = i.weight - self.lr * i.d_weight
                i.bias = i.bias - self.lr * i.d_bias

                if (self.debug is True):
                    print(f"[debug]:weight shape: {i.weight.shape}\n")
            if (self.debug is True):
                    print(f"[debug]:weight shape: {self.layer[-1].weight}\n")
        elif method == 'Adam':
            num_iter += 1
            for i in self.layer:
                i.momentum1 = beta1 * i.momentum1 + (1-beta1) * i.d_weight
                i.momentum2 = beta2 * i.momentum2 + (1-beta2) * i.d_weight * i.d_weight
                momentum1_unbias = i.momentum1 / (1 - beta1 ** num_iter)
                momentum2_unbias  = i.momentum2 / (1 - beta2 ** num_iter)
                i.weight -= self.lr * momentum1_unbias / (np.sqrt(momentum2_unbias) + 1e-8)

    def save_current_weight(self):
        for l in self.layer:
            l.saved_bias = l.bias
            l.saved_weight = l.weight

    def save_to_ckpt(self,o_path:str="./ckpt"):
        for i in range(len(self.layer_size)-1):
            path = o_path
            l = self.layer[i]
            path += f"/layer_{i}/"
            if not os.path.exists(path):
                os.makedirs(path)
            path += "ckpt.npz"
            np.savez(path, weight = l.saved_weight,bias=l.saved_bias)
        
class BasicRoutinizer(BaseModel):
    max_iter:int
    feature_data:np.ndarray
    label_data:np.ndarray
    save_ckpt:Optional[bool] = True
    lr: Optional[float]=1e-5
    beta1: Optional[float]=0.9
    beta2: Optional[float]=0.999
    dynamic_show: Optional[bool] = False
    dynamic_show_res: Optional[int] = 5000
    acc_b:Optional[bool] = False
    
    batchsize:Optional[int] = None

    val_feature_data:Optional[np.ndarray] = None
    val_label_data:Optional[np.ndarray] = None
    
    method: Optional[str]='Adam'
    min_loss:Optional[float] = 1e-2
    max_acc:Optional[float] = 0.8
    type:Optional[str] = 'MSE'
    path:Optional[str] = './ckpt'
    loss=[]
    val_loss=[]
    acc=[]
    val_acc=[]
    final_ans=[]
    
    class Config:
        arbitrary_types_allowed = True

    # Should save a best model!
    def run(self,m:Model):
        m.lr = self.lr
        reach_min_loss = 0
        val = False
        if self.val_feature_data is not None and self.val_label_data is not None:
            val = True
        
        # Rotationally choose the mini-batch
        # rot += self.batchsize at per iter end
        # rot = rot mod N
        # choose feature_data [rot,rot+batchsize]
        
        if self.dynamic_show is True:
            fig = plt.figure()  
            plt.ion()  
        for i in tqdm(range(self.max_iter)):
            rot = 0
            if self.batchsize == None:
                rotN = 1
            else:
                rotN = np.ceil(np.size(self.feature_data,1) / self.batchsize)
            
            for j in range(int(rotN)):
                if self.batchsize == None:
                    feature_data = self.feature_data
                    label_data = self.label_data
                else:          
                    feature_data = self.feature_data.T[rot:rot+self.batchsize].T
                    label_data = self.label_data.T[rot:rot+self.batchsize].T
                    rot = rot+self.batchsize

                predict = m.forward(feature_data)
                if val is True:
                    
                    val_predict = m.forward(self.val_feature_data,train=False)
                    
   
                loss = m.cal_loss(label_data,type=self.type)
                npredict = m.logits2res(predict)
                acc = m.acc(real_data=label_data,predict_data=npredict)
                if val is True:
                    val_npredict = m.logits2res(val_predict)
                    val_acc = m.acc(real_data=self.val_label_data,predict_data=val_npredict)
            
                m.back_propagate()
                m.update(method=self.method,num_iter=i,debug=True,beta1=self.beta1,beta2=self.beta2)
            
                # plot
                self.loss.append(loss)
                self.acc.append(acc)
                if val is True:
                    val_loss = m.cal_loss(self.val_label_data,type=self.type,train=False,val_input=val_predict)
                    self.val_loss.append(val_loss)
                    self.val_acc.append(val_acc)
                if self.save_ckpt is True:
                    if val is True:
                        l = val_loss
                        a = acc
                    else:
                        l = loss
                        a = val_acc
                    #if l < self.min_loss:
                    if a > self.max_acc:
                        # reach_min_loss = 1
                        # self.min_loss = l
                        reach_min_loss = 1
                        self.max_acc = a
                        m.save_current_weight()

            if self.dynamic_show is False and i % self.dynamic_show_res == 0:
                print(f"{i}th iteration: Loss is {loss}")
            if self.dynamic_show is True and i % self.dynamic_show_res == 0:
                fig.clf()  
                fig.suptitle("Loss change with epoch")
                ax = fig.add_subplot(121)
                ax.set_yscale("log")
                ax.scatter(range((i+1)*(int(rotN))), self.loss, facecolor="none", edgecolor='#e474f2', s=2)
                ax.plot(range((i+1)*(int(rotN))), self.loss, color='#e474f2' , label="loss")
                if val is True:
                    ax.scatter(range((i+1)*(int(rotN))), self.val_loss, facecolor="none", edgecolor='#3464f2', s=2)
                    ax.plot(range((i+1)*(int(rotN))), self.val_loss, color='#3464f2' , label="loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                if self.acc_b is True:
                    bx = fig.add_subplot(122)
                    bx.scatter(range((i+1)*(int(rotN))), self.acc, facecolor="none", edgecolor='#e474f2', s=2)
                    bx.plot(range((i+1)*(int(rotN))), self.acc, color='#e474f2' , label="acc")
                    if val is True:
                        bx.scatter(range((i+1)*(int(rotN))), self.val_acc, facecolor="none", edgecolor='#3464f2', s=2)
                        bx.plot(range((i+1)*(int(rotN))), self.val_acc, color='#3464f2' , label="acc")
                    bx.set_xlabel("Epoch")
                    bx.set_ylabel("Acc")
                
                plt.pause(0.2)


            self.final_ans.append(m.final_ans.ravel().tolist())
        
        if self.dynamic_show is True:
            plt.ioff()
            plt.show()

        
        if self.save_ckpt is True:
            
            if reach_min_loss == 0:
                print(f'Never reach at expected loss {self.min_loss}\n')
                m.save_current_weight()

            m.save_to_ckpt(o_path=self.path)
            print("Saved model!")

    def print_loss(self):
        plt.yscale("log")
        plt.scatter(range(self.max_iter), self.loss, facecolor="none", edgecolor='#e474f2', s=2)
        plt.plot(range(self.max_iter), self.loss, color='#e474f2' , label="loss")
        plt.title("Loss change with epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
    

            
            



            

