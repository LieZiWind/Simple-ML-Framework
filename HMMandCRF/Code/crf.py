import pycrfsuite


def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),

    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),

        })
    else:
        features['BOS'] = True 

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),

        })
    else:
        features['EOS'] = True  

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for _, label in sent]

def sent2tokens(sent):
    return [token for token, _ in sent]

def read_data(filename):
        with open(filename, 'r',encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines()]    
        sentences = []
        sentence = []
        for line in lines:
            if line:
                word, tag = line.split()
                sentence.append((word, tag))
            else:
                if sentence: sentences.append(sentence)
                sentence = []
        return sentences
# sentences1 = read_data(r'NER\English\train.txt')
# sentences2 = read_data(r'NER\English\validation.txt')
# X_train = [sent2features(s) for s in sentences1]
# y_train = [sent2labels(s) for s in sentences1]
# X_test = [sent2features(s) for s in sentences2]

# trainer = pycrfsuite.Trainer()

# for xseq, yseq in zip(X_train, y_train):
#     trainer.append(xseq, yseq)

# trainer.set_params({
#     'c1': 1.0,  
#     'c2': 1e-3,  
#     'max_iterations': 100000,  
# })

# trainer.train('crf_eng.model')  

# tagger = pycrfsuite.Tagger()
# tagger.open('crf_eng.model')
# y_pred_eng = [tagger.tag(xseq) for xseq in X_test]
# tagger.close()

# def write_output(data_filename,output_filename,y_pred):
#         with open(data_filename, 'r',encoding='utf-8') as file:
#             lines = [line.strip() for line in file.readlines()]    
#         sentences = []
#         sentence = []
#         for line in lines:
#             if line:
#                 word, tag = line.split()
#                 sentence.append((word, tag))
#             else:
#                 if sentence: sentences.append(sentence)
#                 sentence = []
#         with open(output_filename, 'w',encoding='utf-8') as file:
#             for sentence,tags in zip(sentences,y_pred):
#                 words, _ = zip(*sentence)
#                 for word, tag in zip(words, tags):
#                     file.write(f"{word} {tag}\n")
#                 file.write("\n")

# write_output(r'NER\English\validation.txt',r'CRFpredict.txt',y_pred_eng)


