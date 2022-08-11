from transformers import MT5ForConditionalGeneration, T5Config, AdamW, get_scheduler, BatchEncoding
from tqdm.auto import tqdm
import torch
import os
from random import shuffle
from transformers.trainer_utils import set_seed
from sys import argv
from collections import defaultdict
from transformers.file_utils import to_py_obj
import pickle
from functools import partial
from collections import defaultdict
import re
seed = 100
set_seed(seed)

class DummyTokenizer:
    def __init__(self):
        self.t2id = defaultdict(lambda: len(self.t2id))
        self.t2id.update({'<pad>': 0, '<eos>': 1, '<sep>': 2})
        self.id2t = None
        self.special_idxs = {0, 1, 2}

    def __call__(self, sent):
        sent = self.manipulate_feats(sent)
        sent = [self.t2id[char] for char in sent] + [1]
        mask = [1]*len(sent)

        sent = torch.LongTensor(sent).view(1, -1)
        mask = torch.LongTensor(mask).view(1, -1)

        return BatchEncoding(data={'input_ids': sent, 'attention_mask': mask})

    def __len__(self):
        return len(self.t2id)

    def save_pretrained(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.t2id), f)

    @classmethod
    def from_pretrained(cls, path):
        tok = cls()
        with open(path, 'rb') as f:
            tok.t2id = pickle.load(f)
        return tok

    def inverse_dict(self):
        self.id2t = [None]*len(self.t2id)
        for token, idx in self.t2id.items():
            self.id2t[idx] = token
        assert not any([token is None for token in self.id2t])

    def decode(self, token_ids, skip_special_tokens=False):
        if not self.id2t:
            self.inverse_dict()
        token_ids: list = to_py_obj(token_ids)

        try:
            return [self.id2t[idx] for idx in token_ids if not (skip_special_tokens and idx in self.special_idxs)]
        except TypeError:
            self.inverse_dict()
            return [self.id2t[idx] for idx in token_ids if not (skip_special_tokens and idx in self.special_idxs)]

    @staticmethod
    def manipulate_feats(inp):
        inp = inp.split('<sep>')
        res = []
        for part in inp:
            if part.startswith(';'):
                res += part[1:].split(';')
            else:
                res += list(part)
            res.append('<sep>')
        res = res[:-1]

        return res

def load_dataset(path, task):
    assert task in {'inf', 'reinf', 'analysis'}
    originals = []
    examples = []
    all_feats = set()
    with open(path, encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if not line: continue
            originals.append(line)
            line = [dist_cases(x) for x in line]
            if task == 'reinf':
                input_feats, output_feats = line[0].split(';'), line[2].split(';')
                feats = {';' + feat for feat in input_feats+output_feats}
                inp = line[1] + '<sep>;' + line[0] + '<sep>;' + line[2]
                outp = line[3]
            elif task == 'inf':
                feats = line[1].split(';')
                feats = {';' + feat for feat in feats}
                inp = line[0] + '<sep>;' + line[1]
                outp = line[2]
            else:
                line[1] = line[1].split()
                feats = line[1][-1].split(';')
                feats = {';' + feat for feat in feats}
                inp = line[0]
                outp = ' '.join(line[1][:-1]) + '<sep>;' + line[1][-1]
            all_feats |= feats
            examples.append([inp, outp])
    return originals, examples, all_feats

def tokenize_dataset(examples, tokenizer, mode='train'):
    for example in examples:
        example[0] = tokenizer(example[0])
    if mode == 'train':
        for example in examples:
            example[1] = tokenizer(example[1])
    return examples

pat = re.compile(';[A-Z]+?\+?[A-Z]*?\(.+?\)')
def dist_cases(feats: str):
    res = feats
    for match in pat.finditer(feats):
        text = match.group()
        case = text[1:text.index('(')]
        orig = text[text.index('(')+1:-1].split(',')
        distributed = ';'+';'.join([case + '-' + feat for feat in orig])
        res = res.replace(text, distributed)
    return res

def dedist_cases(feats):
    feats = feats.split(';')
    new = []
    dicto = defaultdict(list)
    for feat in feats:
        if '-' in feat:
            feat = feat.split('-')
            dicto[feat[0]].append(feat[1])
            if new[-1] != feat[0]:
                new.append(feat[0])
        else:
            new.append(feat)
    for case in dicto:
        new[new.index(case)] = f'{case}({",".join(dicto[case])})'
    return ';'.join(new)

num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lang = argv[2]
size = 'xxs'
task = argv[1]
data_dir = task
max_len = 60

cfg = partial(T5Config, feed_forward_proj="gated-gelu", decoder_start_token_id=0)
sizes = {
'xxs': partial(cfg, f_dd=256,
                    num_decoder_layers=1,
                    num_heads=3,
                    num_layers=1,
                    d_model=128),
}

config = sizes[size]

train_path = os.path.join('/kuacc/users/eacikgoz17/NLP/MRL_shared-task_2022', data_dir, f'{lang}.trn')
test_path = os.path.join('/kuacc/users/eacikgoz17/NLP/MRL_shared-task_2022', data_dir, f'{lang}.dev')
save_path = os.path.join('predictions', 'transformer', f'{lang}_{task}_{size}')

_, train_dataset, morpho_feats = load_dataset(train_path, task)
original_test, test_dataset, _ = load_dataset(test_path, task)

if os.path.isdir(save_path):
    model = MT5ForConditionalGeneration.from_pretrained(save_path).cuda()
    tokenizer = DummyTokenizer.from_pretrained(os.path.join(save_path, 'tok.pkl'))
    test_dataset = tokenize_dataset(test_dataset, tokenizer, mode='test')
    print('model loaded!')
else:
    tokenizer = DummyTokenizer()
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, mode='test')
    print('data loaded!')


    model = MT5ForConditionalGeneration(config(vocab_size=len(tokenizer))).cuda()
    print('model loaded!')

    num_training_steps = num_epochs * len(train_dataset)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        shuffle(train_dataset)
        for example in train_dataset:
            example = [sent.to(device) for sent in example]
            outputs = model(**example[0], labels=example[1]["input_ids"])
            loss = outputs.loss

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(os.path.join(save_path, 'tok.pkl'))

model.eval()
tot = len(test_dataset)
corrects = 0
out_path = os.path.join('predictions', 'mTnew', task, f'{lang}.dev')
out_file = "/kuacc/users/eacikgoz17/NLP/MRL_shared-task_2022/predictions/mTnew/" + task + "/out/" + lang + ".dev"
out = open(out_path, 'w', encoding='utf8')
outP = open(out_file, 'w', encoding='utf8')
for i, (example, target) in enumerate(test_dataset):
    example = example.to(device)
    with torch.no_grad():
        beam_output = model.generate(example.input_ids, max_length=max_len, num_beams=5, early_stopping=True)
        pred = tokenizer.decode(beam_output[0], skip_special_tokens=True)
        if task == 'analysis':
            p = [False] + [c.isupper() or c.isnumeric() for c in pred[1:]]
            try:
                p = p.index(True)
                pred = [''.join(pred[:p])] + pred[p:]
                pred[0] += '<sep>'
                pred = ';'.join(pred)
                pred = dedist_cases(pred)
                pred = pred.replace('<sep>;', ' ')
            except ValueError:
                pass
        else:
            pred = ''.join(pred)
    print(pred, original_test[i][-1])
    outP.write(pred+"\n")
    out.write('\t'.join(original_test[i][:-1]) + '\t' + pred + '\n')
    if pred == original_test[i][-1]:
        corrects += 1
outP.close()
print(corrects)
