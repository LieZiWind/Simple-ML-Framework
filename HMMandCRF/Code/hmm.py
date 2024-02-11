import numpy as np,json

class HMM():
    """Hidden Markov Model
    Directly count the hidden-state - observation sequence to estimate the pi, transfer and  fire matrix.
    Input file in jsonl, per line is {"word":[..., ..., ...],"label":[..., ..., ...]}
    """
    def __init__(self,path:str="English-train.jsonl"):
        self.word_index = {}
        self.label_index = {}
        self.state_num = 0
        self.observation_num = 0
        self.pi = None
        self.transfer = None
        self.emission = None
        with open(path,"r") as src:
            lines = list(src)
            self.sequences = [None for _ in lines]
            for idx,line in enumerate(lines):
                self.sequences[idx] = json.loads(line)
    
    def _build_index_system(self):
        for seq in self.sequences:
            for word in seq["word"]:
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)
            for label in seq["label"]:
                if label not in self.label_index:
                    self.label_index[label] = len(self.label_index)
    
    def _build_matrix(self,epsilon=1e-8):
        self.observation_num = len(self.word_index)
        self.state_num = len(self.label_index)
        self.pi = np.zeros(self.state_num)
        self.transfer = np.zeros((self.state_num,self.state_num))
        self.emission = np.zeros((self.state_num,self.observation_num))
        for seq in self.sequences:
            previous_label = None
            for idx,word in enumerate(seq["word"]):
                label = seq["label"][idx]
                self.emission[self.label_index[label],self.word_index[word]] += 1
                if idx == 0:
                    self.pi[self.label_index[label]] += 1
                else:
                    self.transfer[self.label_index[previous_label],self.label_index[label]] += 1
                previous_label = label
        # Avoid overflow
        self.pi[self.pi == 0] = epsilon
        self.pi /= self.pi.sum()
        self.transfer[self.transfer == 0] = epsilon
        self.transfer /= self.transfer.sum(axis=1, keepdims=True)
        self.emission[self.emission == 0] = epsilon
        self.emission /= self.emission.sum(axis=1, keepdims=True)

    def train(self):
        self._build_index_system()
        self._build_matrix()
        # For viterbi convenience
        self.pi = np.log(self.pi)
        self.transfer = np.log(self.transfer)
        self.emission = np.log(self.emission)
    
    def _viterbi(self,seq):
        length = len(seq["word"])
        T1_table = np.zeros([length, self.state_num])
        T2_table = np.zeros([length, self.state_num])
        initial_state = self._get_state(seq["word"][0])
        T1_table[0, :] = self.pi + initial_state
        T2_table[0, :] = np.nan
        for i in range(1, length):
            state = self._get_state(seq["word"][i])
            state = np.expand_dims(state, axis=0)
            prev_score = np.expand_dims(T1_table[i-1, :], axis=-1)
            score = prev_score + self.transfer + state
            T1_table[i, :] = np.max(score, axis=0)
            T2_table[i, :] = np.argmax(score, axis=0)
        best_label = int(np.argmax(T1_table[-1, :]))
        best_labels = [best_label]
        for i in range(length-1, 0, -1):
            best_label = int(T2_table[i, best_label])
            best_labels.append(best_label)
        return list(reversed(best_labels))

    def _get_state(self,word):
        idx = self.word_index.get(word,0)
        if idx == 0:
            return np.log(np.ones(self.state_num)/self.state_num)
        return np.ravel(self.emission[:, idx])
    
    def decode(self,dict):
        idx_index = {v: k for k, v in self.label_index.items()}
        if len(dict) == 0:
            raise NotImplementedError("Null input!")
        best_label = self._viterbi(dict)

        ret = []
        for word, idx in zip(dict["word"], best_label):
            ret.append((word,idx_index[idx]))
        
        return ret
     
    
    def predict(self,path="English-validation.jsonl",tp="test.txt"):
        with open(path,"r") as src:
            lines = list(src)
            for line in lines:
                seq = json.loads(line)
                ret = self.decode(seq)
                with open(tp,'a',encoding='utf-8') as ter:
                    for word,label in ret:
                        ter.write(f"{word} {label}\n")
                    ter.write("\n")





# h = HMM(path="English-train.jsonl")
# h.train()
# h.predict(path="English-validation.jsonl",tp="English_result.txt")

# h = HMM(path="Chinese-train.jsonl")
# h.train()
# h.predict(path="Chinese-validation.jsonl",tp="Chinese_result.txt")