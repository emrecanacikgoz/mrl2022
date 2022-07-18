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
            src_tags, src, tgt_tags, tgt = line[0], line[1], line[2], line[3]

            # Create Sperator for word modification
            src = src + ";"

            source_tags  = src_tags.split(self.tag_seperator)
            source_lemma = [char for char in src]
            target_tags  = tgt_tags.split(self.tag_seperator)
            target       = [char for char in tgt]

            source = source_tags + source_lemma + target_tags
            
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

        default           = {pad_token : 0, start_token : 1, eos_token : 2, unk_token : 3}
        source_encoder    = dict(**default); source_decoder = dict();
        source_decoder[0] = pad_token; source_decoder[1] = start_token; source_decoder[2] = eos_token; source_decoder[3] = unk_token;
        target_encoder    = dict(**default); target_decoder = dict();
        target_decoder[0] = pad_token; target_decoder[1] = start_token; target_decoder[2] = eos_token; target_decoder[3] = unk_token;

        sources, targets = [], []
        for sample in data:
            sources.extend(sample[0])
            targets.extend(sample[1])  
        
        for j, tag in enumerate(list(set(targets))):
            target_encoder[tag] = j+4
            target_decoder[j+4] = tag
            
        for j, surf in enumerate(list(set(sources))):
            source_encoder[surf] = j+4
            source_decoder[j+4]  = surf

        self.source_encoder, self.source_decoder = source_encoder, source_decoder
        self.target_encoder, self.target_decoder = target_encoder, target_decoder

        #print(f"Source Encoder: {self.source_encoder}")
        #print(f"Source Decoder: {self.source_decoder}")
        #print(f"Target Encoder: {self.target_encoder}")
        #print(f"Target Decoder: {self.target_decoder}\n")
        
        self.data = data

    def encode(self, x):

        src = []
        for i in self.handle_input(x[0]):
            if i in self.source_encoder:
                src.append(self.source_encoder[i])
            else:
                src.append(self.source_encoder['<unk>'])

        tgt = []
        for i in self.handle_input(x[1]):
            if i in self.target_encoder:
                tgt.append(self.target_encoder[i])
            else:
                tgt.append(self.target_encoder['<unk>'])

        src, tgt = torch.tensor(src), torch.tensor(tgt)

        return src, tgt

    def decode(self, x):
        return [self.source_decoder[i] for i in x[0]], [self.target_decoder[i] for i in x[1]]

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



