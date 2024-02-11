import json

def better_organize(path:str="NER/English/train.txt",terminal_path:str="English-train.jsonl"):
    with open(path,"r",encoding='utf-8') as f:
        
        new_data_struct=[]
        sequence={"word":[],"label":[]}
        for idx,line in enumerate(f):
            if line == "\n":
                new_data_struct.append(sequence)
                sequence={"word":[],"label":[]}
            else:
                word,label = line.split(" ")
                sequence["word"].append(word)
                sequence["label"].append(label.rstrip())

    with open(terminal_path,"a") as j:
        for seq in new_data_struct:
            json.dump(seq,j)
            j.write('\n')




# #better_organize(path="NER/Chinese/train.txt",terminal_path="Chinese-train.jsonl")
# better_organize(path="NER/English/validation.txt",terminal_path="English-validation.jsonl")
        