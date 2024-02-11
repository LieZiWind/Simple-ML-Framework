import json
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(1)
START_LABEL = "<START>"
STOP_LABEL = "<STOP>"
EMBEDDING_DIM = 128
HIDDEN_DIM = 128

def argmax(vec):
    _, idx = torch.max(vec,axis=1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class TRANSFORMER_CRF(nn.Module):
    def __init__(self, sequences,embedding_dim, hidden_dim,nhead:int=4,num_layers:int=1):
        super(TRANSFORMER_CRF, self).__init__()
        self.sequences = sequences
        self.word_index = {}
        self.label_index = {}
        self.num_word = 0
        self.num_label = 0
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self._build_index_system()
        self.word_embeds = nn.Embedding((self.num_word+1), self.embedding_dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embedding_dim,nhead=nhead,dim_feedforward=self.hidden_dim
        ), num_layers=num_layers)
        self.hidden2tag = nn.Linear(self.embedding_dim, self.num_label)
        self.transitions = nn.Parameter(torch.randn(self.num_label, self.num_label))
        self.transitions.data[self.label_index[START_LABEL], :] = -10000
        self.transitions.data[:, self.label_index[STOP_LABEL]] = -10000

        self.hidden = self.init_hidden()
    def init_hidden(self):
        h0 = torch.randn(2, 1, self.hidden_dim//2)
        c0 = torch.randn(2, 1, self.hidden_dim//2)
        return (h0, c0)

    def _get_transformer_features(self,sentence):
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        transformer_out = self.transformer(embeds)
        transformer_out = transformer_out.view(len(sentence), self.embedding_dim)
        feats = self.hidden2tag(transformer_out)
        return feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.label_index[START_LABEL]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.label_index[STOP_LABEL], tags[-1]]
        return score

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.num_label), -10000.)
        init_alphas[0][self.label_index[START_LABEL]] = 0.
        previous = init_alphas
        for obs in feats:
            alphas_t = []
            for next_tag in range(self.num_label):
                emit_score = obs[next_tag].view(1, -1).expand(1, self.num_label)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = previous + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            previous = torch.cat(alphas_t).view(1, -1)
        terminal_var = previous + self.transitions[self.label_index[STOP_LABEL]]
        scores = log_sum_exp(terminal_var)
        return scores


    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.num_label), -10000.)
        init_vvars[0][self.label_index[START_LABEL]] = 0

        previous = init_vvars
        for obs in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.num_label):
                next_tag_var = previous + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            previous = (torch.cat(viterbivars_t) + obs).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = previous + self.transitions[self.label_index[STOP_LABEL]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.label_index[START_LABEL]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_transformer_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score


    def forward(self, sentence):
        feats = self._get_transformer_features(sentence)
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
    
    def _build_index_system(self):
        for seq in self.sequences:
            for word in seq["word"]:
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)
            for label in seq["label"]:
                if label not in self.label_index:
                    self.label_index[label] = len(self.label_index)
        self.label_index[START_LABEL] = len(self.label_index)
        self.label_index[STOP_LABEL] = len(self.label_index)
        self.num_label = len(self.label_index)
        self.num_word = len(self.word_index)
    
    def prepare_sequence(self,seq):
        idxs = [None for _ in seq]
        for i,w in enumerate(seq):
            idx = self.word_index.get(w,-1)+1
            idxs[i] = idx
        return torch.tensor(idxs, dtype=torch.long)
    





