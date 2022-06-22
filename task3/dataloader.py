import torch
from torch.utils.data import Dataset


class Parser:
    def __init__(self, part_seperator="\t", tag_seperator=";"):
        
        self.part_seperator = part_seperator
        self.tag_seperator  = tag_seperator 

    def parse_file(self, file):

        data = []
        for line in open(file):

            line =line.rstrip().split(self.part_seperator)
            src, tgt = line[0], line[1].replace(" ", self.tag_seperator).split(self.tag_seperator)
            source = [char for char in src]
            #print(f"Source: {src} ===> Target: {tgt}")
            target = [char for char in tgt[0]] + tgt[1:]
            #print(f"Source: {source} ===> Target: {target}\n")
            data.append([source, target])
            
        return data
            

class Vocab:
    def __init__(self, data, pad_to=-1, start_token="<s>", eos_token="</s>", pad_token="<p>"):

        self.pad_to      = pad_to
        self.start_token = start_token
        self.eos_token   = eos_token
        self.pad_token   = pad_token

        default         = {pad_token : 0, start_token : 1 , eos_token : 2}
        surf_encoder    = dict(**default); surf_decoder = dict();
        surf_decoder[0] = pad_token; surf_decoder[1] = start_token; surf_decoder[2] = eos_token;
        feat_encoder    = dict(**default); feat_decoder = dict();
        feat_decoder[0] = pad_token; feat_decoder[1] = start_token; feat_decoder[2] = eos_token;

        lemmas, tags = [], []
        for sentence in data:
            lemmas.extend(sentence[0])
            tags.extend(sentence[1])  
        
        for j, tag in enumerate(list(set(tags))):
            feat_encoder[tag] = j+3
            feat_decoder[j+3] = tag
            
        for j, surf in enumerate(list(set(lemmas))):
            surf_encoder[surf] = j+3
            surf_decoder[j+3]  = surf

        self.surf_encoder, self.surf_decoder = surf_encoder, surf_decoder
        self.feat_encoder, self.feat_decoder = feat_encoder, feat_decoder
        
        self.data = data

    def encode(self, x):

        src = []
        for i in self.handle_input(x[0]):
            src.append(self.surf_encoder[i])

        tgt = []
        for i in self.handle_input(x[1]):
            tgt.append(self.feat_encoder[i])

        src = torch.tensor(src)
        tgt = torch.tensor(tgt)

        return src, tgt

    def decode(self, x):
        return [self.surf_decoder[i] for i in x[0]], [self.feat_decoder[i] for i in x[1]]


    def handle_input(self, x):

        right_padding = self.pad_to - len(x) - 2
        return [self.start_token] + x + [self.eos_token] + [self.pad_token] * right_padding 

        
class WordLoader(Dataset):
    def __init__(self, data, pad_to=-1, start_token="<s>", eos_token="</s>", pad_token="<p>"):
        assert pad_to != -1
        
        self.vocab  = Vocab(data, pad_to=pad_to, start_token=start_token, eos_token=eos_token, pad_token=pad_token)
        self.data   = data

    def __getitem__(self, idx):
        return self.vocab.encode(self.data[idx])
            
    def __len__(self):
        return len(self.data)



