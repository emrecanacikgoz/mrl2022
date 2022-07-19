import torch
from torch.utils.data import Dataset, DataLoader


class Parser:
    def __init__(self, part_seperator="\t", tag_seperator=";"):
        
        self.part_seperator = part_seperator
        self.tag_seperator  = tag_seperator 

    def parse_file(self, file):

        data = []
        for line in open(file):

            line =line.rstrip().split(self.part_seperator)
            src_lemma, src_feat, tgt = line[0], line[1], line[2]
            #print(f"Source Lemma: {src_lemma}, Source Feature: {src_feat}, Target: {tgt}\n")

            source_lemma = [char for char in src_lemma]
            source_feat  = src_feat.split(self.tag_seperator)

            source = source_lemma + source_feat
            target = [char for char in tgt]
            
            #print(f"Source: {source}, Target: {target}\n")
            data.append([source, target])
        return data
            

class Vocab:
    def __init__(self, data, pad_to=-1, start_token="<s>", eos_token="</s>", pad_token="<p>",  unk_token="<unk>"):

        self.pad_to      = pad_to
        self.start_token = start_token
        self.eos_token   = eos_token
        self.pad_token   = pad_token
        self.unk_token   = unk_token

        default         = {pad_token : 0, start_token : 1, eos_token : 2, unk_token : 3}
        surf_encoder    = dict(**default); surf_decoder = dict();
        surf_decoder[0] = pad_token; surf_decoder[1] = start_token; surf_decoder[2] = eos_token; surf_decoder[3] = unk_token;
        feat_encoder    = dict(**default); feat_decoder = dict();
        feat_decoder[0] = pad_token; feat_decoder[1] = start_token; feat_decoder[2] = eos_token; feat_decoder[3] = unk_token;

        lemmas, tags = [], []
        for sentence in data:
            lemmas.extend(sentence[1])
            tags.extend(sentence[0])  
        
        for j, tag in enumerate(list(set(tags))):
            feat_encoder[tag] = j+4
            feat_decoder[j+4] = tag
            
        for j, surf in enumerate(list(set(lemmas))):
            surf_encoder[surf] = j+4
            surf_decoder[j+4]  = surf

        self.surf_encoder, self.surf_decoder = surf_encoder, surf_decoder
        self.feat_encoder, self.feat_decoder = feat_encoder, feat_decoder

        #print(f"Feature Encoder: {self.feat_encoder}")
        #print(f"Surface Decoder: {self.feat_decoder}")
        #print(f"Surface Encoder: {self.surf_encoder}")
        #print(f"Surface Decoder: {self.surf_decoder}\n")
        
        self.data = data

    def encode(self, x):

        src = []
        
        for i in self.handle_input(x[0]):
            if i in self.feat_encoder:
                src.append(self.feat_encoder[i])
            else:
                src.append(self.feat_encoder['<unk>'])

        tgt = []
        for i in self.handle_input(x[1]):
            if i in self.surf_encoder:
                tgt.append(self.surf_encoder[i])
            else:
                tgt.append(self.surf_encoder['<unk>'])

        src, tgt = torch.tensor(src), torch.tensor(tgt)
        
        return src, tgt

    def decode(self, x):
        return [self.feat_decoder[i] for i in x[0]], [self.surf_decoder[i] for i in x[1]]


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



