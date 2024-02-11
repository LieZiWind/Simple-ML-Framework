from pydantic import BaseModel
import numpy as np
from typing import Optional,List,Any

class Data(BaseModel):
    """
    y_train: Shape (,N)
    X_train: Shape (F,N)
    """
    y_train:np.ndarray
    X_train:np.ndarray
    y_val:Optional[np.ndarray] = None
    X_val:Optional[np.ndarray] = None
    class Config:
        arbitrary_types_allowed = True

    def shuffle(self,seed:int=42):
        np.random.seed(seed)
        np.random.shuffle(self.y_train.T)
        np.random.seed(seed)
        np.random.shuffle(self.X_train.T)
    def normalize(self,seed:int=42):
        mean_X_train = np.mean(self.X_train,1).reshape(-1,1)
        std_X_train = np.std(self.X_train,1).reshape(-1,1)
        

        # Now we deal with std = 0 situation
        std_X_train = np.where(std_X_train == 0,1e-12,std_X_train)
        np.savez("meanstd.npz", mean = mean_X_train,std=std_X_train)
        
        self.X_train = (self.X_train - mean_X_train) / std_X_train

        if self.X_val is not None:
            self.X_val = (self.X_val - mean_X_train) / std_X_train

    def create_val_set(self,seed:int=42):
        N = np.size(self.X_train,1)
        train_N = N - N // 5
        print(train_N)
        print(self.y_train.shape)

        # Before splitting the val and train set, we need to randomly shuffle the original data
        np.random.seed(seed)
        np.random.shuffle(self.y_train.T)
        np.random.seed(seed)
        np.random.shuffle(self.X_train.T)

        # Now we split them.
        self.y_train, self.y_val = np.split(self.y_train, [train_N],1)
        self.X_train, self.X_val = np.split(self.X_train, [train_N],1)

    def create_k_cross_val_set(self):
        pass

