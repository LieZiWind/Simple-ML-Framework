class LegacyHMM():
    """
    Deprecated. This class is used when we can only get the observation.
    Using Baum-Welch algorithm (a version of EM) to estimate the pi, transfer and fire matrix.
    """
    """Hidden Markov Model
    transfer is seen as a matrix, pi is seen as a column vector (p)
    Every col of transfer is a prob (p' = Tp)
    Every col of fire is a prob (O = Fp)
    """
    def __init__(self,state_num:int,observation_num:int):
        temp = np.random.uniform(0,1,size=(state_num,1))
        self.pi = temp/temp.sum()
        temp = np.random.uniform(0,1,size=(state_num,state_num))
        self.transfer = temp/temp.sum(axis=0, keepdims=True)
        temp = np.random.uniform(0,1,size=(observation_num,state_num))
        self.fire = temp/temp.sum(axis=0, keepdims=True)
        self.observation_num = observation_num
        self.state_num = state_num
    def __str__(self) -> str:
        return f"Initial distribution is:\n{self.pi}\nTransfer Prob Matrix is:\n{self.transfer}\nObervation fire matrix is:\n{self.fire}"
    
    def train(self,list_of_seq:list,max_iter:int=10):
        for i in range(max_iter):
            self.update(list_of_seq=list_of_seq)

    def build_index_system(self,list_of_seq:list):
        pass
    def index_system(self,name:str):
        return 0

    def update(self,list_of_seq:list):
        pi_star = np.zeros((self.state_num,1))
        transfer_star_1 = np.zeros((self.state_num,self.state_num))
        transfer_star_2 = np.zeros((self.state_num,1))
        fire = np.zeros((self.observation_num,self.state_num))
        fire_2 = np.zeros((self.state_num,1))
        for idx,seq in enumerate(list_of_seq):
            gamma_list,epsilon_list = self.seq(seq=seq)
            pi_star += gamma_list[0]
            in_transfer_star_1 = np.zeors((self.state_num,self.state_num))
            in_transfer_star_2 = np.zeros((self.state_num,1))
            in_fire_2 = np.zeros((self.state_num,1))
            for j,gamma in enumerate(gamma_list):
                if j <= len(gamma_list)-2:
                    in_transfer_star_2 += gamma
                    in_transfer_star_1 += epsilon_list[j]
                in_fire_2 += gamma
                word = seq["word"][j]
                observe = self.index_system(word)
                fire[observe] += gamma.reshape(1,-1)
            transfer_star_1 += in_transfer_star_1
            transfer_star_2 += in_transfer_star_2
            fire_2 += in_fire_2
        self.pi = pi_star / (idx+1)
        self.transfer = transfer_star_1 / transfer_star_2
        self.fire = fire / fire_2.T

    def seq(self,seq:dict):
        text = seq["word"]
        l = len(text)
        alpha = None
        beta = None
        alpha_list=[]
        beta_list=[]
        gamma_list=[]
        epsilon_list=[]
        for idx in range(l):
            alpha = self._recursive_forward(text_seq=text,alpha=alpha,t=idx)
            alpha_list.append(alpha)
            beta = self._recursive_backward(text_seq=text,beta=beta,t=idx)
            beta_list.append(beta)
        for idx in range(l):
            alpha = alpha_list[idx]
            beta = beta_list[l-idx-1]
            gamma  = alpha * beta
            gamma = gamma / np.sum(gamma)
            gamma_list.append(gamma)
            if idx <= l-2:
                beta2 = beta_list[l-idx-2]
                observe = self.index_system(text[idx+1])
                inter = beta2 * self.fire[observe].reshape(-1,1)
                inter = alpha @ inter.T
                epsilon = inter * self.transfer
                epsilon = epsilon / np.sum(epsilon)
                epsilon_list.append(epsilon)
        return gamma_list,epsilon_list
    
    def _recursive_forward(self,text_seq,alpha=None,t=0):
        """t begins at 0
        """
        if t == 0:
            observe = self.index_system(text_seq[t])
            alpha = self.pi*self.fire[observe].reshape(-1,1)
            return alpha
        else:
            observe = self.index_system(text_seq[t])
            inter = self.transfer @ alpha
            alpha = inter*self.fire[observe].reshape(-1,1)
            return alpha
    
    def _recursive_backward(self,text_seq,beta=None,t=0):
        """t begins at 0
        """
        l = len(text_seq)
        if t == 0:
            beta = np.ones((self.state_num,1))
            return beta
        else:
            observe = self.index_system(text_seq[l-t-1])
            inter = self.fire[observe].reshape(-1,1) * beta
            beta = self.transfer @ inter
            return beta
