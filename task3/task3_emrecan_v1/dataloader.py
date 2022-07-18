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
            src, tgt = line[0], line[1]
            space_num = tgt.count(" ")
            #src, tgt = line[0], line[1].replace(" ", self.tag_seperator).split(self.tag_seperator)
            src, tgt = line[0], line[1]
            idx = tgt.rfind(" ")
            tgt_lemma, tgt_feat =  [char for char in tgt[:idx]], tgt[idx:].split(self.tag_seperator)
            target = tgt_lemma + tgt_feat

            source = [char for char in src]
            """target = []
            for lemma in tgt[:space_num]:
                lemmas = [char for char in lemma]
                target.append(lemmas)

            if space_num == 2:
                target[0].extend(" ")

            lemma_tgt = [char for lemma in target for char in lemma]
            target    = lemma_tgt + tgt[space_num:]
            print([source, target])
            #breakpoint()"""
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
            lemmas.extend(sentence[0])
            tags.extend(sentence[1])  
        
        for j, tag in enumerate(list(set(tags))):
            feat_encoder[tag] = j+4
            feat_decoder[j+4] = tag
            
        for j, surf in enumerate(list(set(lemmas))):
            surf_encoder[surf] = j+4
            surf_decoder[j+4]  = surf

        self.surf_encoder, self.surf_decoder = surf_encoder, surf_decoder
        self.feat_encoder, self.feat_decoder = feat_encoder, feat_decoder
        #print(f"Surface Encoder: {self.surf_encoder}")
        #print(f"Surface Decoder: {self.surf_decoder}")
        #print(f"Feature Encoder: {self.feat_encoder}")
        #print(f"Surface Decoder: {self.feat_decoder}")

        #self.surf_encoder = {'<p>': 0, '<s>': 1, '</s>': 2, 'g': 3, 'n': 4, 'ş': 5, '?': 6, 'a': 7, 'ç': 8, 'i': 9, 'u': 10, 'm': 11, 's': 12, '!': 13, 'o': 14, 'l': 15, 'v': 16, 'b': 17, 'ı': 18, 'ü': 19, 'h': 20, 'y': 21, 'c': 22, 'd': 23, '.': 24, 'r': 25, 'z': 26, 'ö': 27, ' ': 28, 'p': 29, 't': 30, 'ğ': 31, 'k': 32, 'e': 33, 'f': 34}
        #self.surf_decoder = {0: '<p>', 1: '<s>', 2: '</s>', 3: 'g', 4: 'n', 5: 'ş', 6: '?', 7: 'a', 8: 'ç', 9: 'i', 10: 'u', 11: 'm', 12: 's', 13: '!', 14: 'o', 15: 'l', 16: 'v', 17: 'b', 18: 'ı', 19: 'ü', 20: 'h', 21: 'y', 22: 'c', 23: 'd', 24: '.', 25: 'r', 26: 'z', 27: 'ö', 28: ' ', 29: 'p', 30: 't', 31: 'ğ', 32: 'k', 33: 'e', 34: 'f'}
        #self.feat_encoder = {'<p>': 0, '<s>': 1, '</s>': 2, 'LOC(3,PL)': 3, 'NOM(2,PL)': 4, 'n': 5, 'ABL(1,PL)': 6, 'ACC(2,PL)': 7, 'ABL(3,SG)': 8, 'DAT(3,SG,RFLX)': 9, 'i': 10, 'u': 11, 'm': 12, 'PROG': 13, 'LOC(1,SG,RFLX)': 14, 'LOC(2,SG)': 15, 'l': 16, 'DAT(2,PL)': 17, 'BEN(1,PL)': 18, 'COM(3,SG,RFLX)': 19, 'NEC': 20, 'ABL(3,SG,RFLX)': 21, 'NOM(3,SG)': 22, 'ABL(3,PL,RFLX)': 23, 'INFR': 24, 'COM(1,SG)': 25, 'ABL(2,PL)': 26, 'p': 27, 'LOC(3,SG,RFLX)': 28, 'ğ': 29, 'DAT(3,PL)': 30, 'DAT(3,SG)': 31, 'BEN(2,PL,RFLX)': 32, 'f': 33, 'PST': 34, 'HAB': 35, 'ABL(2,PL,RFLX)': 36, 'a': 37, 'NOM(1,PL)': 38, 'COM(2,PL,RFLX)': 39, 'BEN(3,PL)': 40, 'v': 41, 'b': 42, 'ı': 43, 'ACC(3,PL,RFLX)': 44, 'BEN(3,SG)': 45, 'ACC(1,SG)': 46, 'h': 47, 'Q': 48, 'ACC(3,PL)': 49, 'BEN(2,SG)': 50, 'd': 51, 'LOC(1,PL)': 52, 'ABL(3,PL)': 53, 'COM(3,SG)': 54, 'LOC(2,PL,RFLX)': 55, 'ö': 56, 'DAT(1,PL)': 57, 'DAT(3,PL,RFLX)': 58, 'COM(1,PL)': 59, 'ABL(1,SG)': 60, 'COM(3,PL)': 61, 'k': 62, 'IMP': 63, 'DAT(1,SG,RFLX)': 64, 'g': 65, 'BEN(3,PL,RFLX)': 66, 'PRS': 67, 'COM(2,SG)': 68, 'DAT(2,SG)': 69, 'LOC(2,PL)': 70, 'ACC(2,PL,RFLX)': 71, 'LOC(2,SG,RFLX)': 72, 'o': 73, 'DAT(1,PL,RFLX)': 74, 'COM(1,SG,RFLX)': 75, 'ABL(1,SG,RFLX)': 76, 'BEN(2,PL)': 77, 'ü': 78, 'COM(2,PL)': 79, 'c': 80, 'r': 81, 'NEG': 82, 'ABL(2,SG)': 83, 'ACC(1,PL)': 84, 'DAT(2,SG,RFLX)': 85, 'PERF': 86, 't': 87, 'ACC(2,SG,RFLX)': 88, 'LOC(3,PL,RFLX)': 89, 'e': 90, 'NOM(1,SG)': 91, 'FUT': 92, 'LOC(1,SG)': 93, 'COM(2,SG,RFLX)': 94, 'ş': 95, 'ç': 96, 'NOM(2,SG)': 97, 'ABL(2,SG,RFLX)': 98, 'ABL(1,PL,RFLX)': 99, 'ACC(3,SG)': 100, 'ACC(1,SG,RFLX)': 101, 's': 102, 'IND': 103, 'PRSP': 104, 'DAT(1,SG)': 105, 'BEN(1,SG)': 106, 'y': 107, 'ACC(2,SG)': 108, 'z': 109, 'ACC(3,SG,RFLX)': 110, 'DAT(2,PL,RFLX)': 111, ' ': 112, 'COM(1,PL,RFLX)': 113, 'NOM(3,PL)': 114, 'COM(3,PL,RFLX)': 115, 'LGSPEC1': 116, 'LOC(3,SG)': 117, 'ACC(1,PL,RFLX)': 118, 'NOM(2,PL,LGSPEC2)': 119}
        #self.feat_decoder = {0: '<p>', 1: '<s>', 2: '</s>', 3: 'LOC(3,PL)', 4: 'NOM(2,PL)', 5: 'n', 6: 'ABL(1,PL)', 7: 'ACC(2,PL)', 8: 'ABL(3,SG)', 9: 'DAT(3,SG,RFLX)', 10: 'i', 11: 'u', 12: 'm', 13: 'PROG', 14: 'LOC(1,SG,RFLX)', 15: 'LOC(2,SG)', 16: 'l', 17: 'DAT(2,PL)', 18: 'BEN(1,PL)', 19: 'COM(3,SG,RFLX)', 20: 'NEC', 21: 'ABL(3,SG,RFLX)', 22: 'NOM(3,SG)', 23: 'ABL(3,PL,RFLX)', 24: 'INFR', 25: 'COM(1,SG)', 26: 'ABL(2,PL)', 27: 'p', 28: 'LOC(3,SG,RFLX)', 29: 'ğ', 30: 'DAT(3,PL)', 31: 'DAT(3,SG)', 32: 'BEN(2,PL,RFLX)', 33: 'f', 34: 'PST', 35: 'HAB', 36: 'ABL(2,PL,RFLX)', 37: 'a', 38: 'NOM(1,PL)', 39: 'COM(2,PL,RFLX)', 40: 'BEN(3,PL)', 41: 'v', 42: 'b', 43: 'ı', 44: 'ACC(3,PL,RFLX)', 45: 'BEN(3,SG)', 46: 'ACC(1,SG)', 47: 'h', 48: 'Q', 49: 'ACC(3,PL)', 50: 'BEN(2,SG)', 51: 'd', 52: 'LOC(1,PL)', 53: 'ABL(3,PL)', 54: 'COM(3,SG)', 55: 'LOC(2,PL,RFLX)', 56: 'ö', 57: 'DAT(1,PL)', 58: 'DAT(3,PL,RFLX)', 59: 'COM(1,PL)', 60: 'ABL(1,SG)', 61: 'COM(3,PL)', 62: 'k', 63: 'IMP', 64: 'DAT(1,SG,RFLX)', 65: 'g', 66: 'BEN(3,PL,RFLX)', 67: 'PRS', 68: 'COM(2,SG)', 69: 'DAT(2,SG)', 70: 'LOC(2,PL)', 71: 'ACC(2,PL,RFLX)', 72: 'LOC(2,SG,RFLX)', 73: 'o', 74: 'DAT(1,PL,RFLX)', 75: 'COM(1,SG,RFLX)', 76: 'ABL(1,SG,RFLX)', 77: 'BEN(2,PL)', 78: 'ü', 79: 'COM(2,PL)', 80: 'c', 81: 'r', 82: 'NEG', 83: 'ABL(2,SG)', 84: 'ACC(1,PL)', 85: 'DAT(2,SG,RFLX)', 86: 'PERF', 87: 't', 88: 'ACC(2,SG,RFLX)', 89: 'LOC(3,PL,RFLX)', 90: 'e', 91: 'NOM(1,SG)', 92: 'FUT', 93: 'LOC(1,SG)', 94: 'COM(2,SG,RFLX)', 95: 'ş', 96: 'ç', 97: 'NOM(2,SG)', 98: 'ABL(2,SG,RFLX)', 99: 'ABL(1,PL,RFLX)', 100: 'ACC(3,SG)', 101: 'ACC(1,SG,RFLX)', 102: 's', 103: 'IND', 104: 'PRSP', 105: 'DAT(1,SG)', 106: 'BEN(1,SG)', 107: 'y', 108: 'ACC(2,SG)', 109: 'z', 110: 'ACC(3,SG,RFLX)', 111: 'DAT(2,PL,RFLX)', 112: ' ', 113: 'COM(1,PL,RFLX)', 114: 'NOM(3,PL)', 115: 'COM(3,PL,RFLX)', 116: 'LGSPEC1', 117: 'LOC(3,SG)', 118: 'ACC(1,PL,RFLX)', 119: 'NOM(2,PL,LGSPEC2)'}
        
        self.data = data

    def encode(self, x):

        src = []
        for i in self.handle_input(x[0]):
            if i in self.surf_encoder:
                src.append(self.surf_encoder[i])
            else:
                src.append(self.surf_encoder['<unk>'])

        tgt = []
        for i in self.handle_input(x[1]):
            if i in self.feat_encoder:
                tgt.append(self.feat_encoder[i])
            else:
                tgt.append(self.feat_encoder['<unk>'])


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



