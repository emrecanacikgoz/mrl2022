from transformers import MT5ForConditionalGeneration, T5Tokenizer, AdamW, get_scheduler
from tqdm.auto import tqdm
import torch
import os
from random import shuffle
from transformers.trainer_utils import set_seed
import re
from collections import defaultdict
from sys import argv
seed = 100
set_seed(seed)

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
        example[0] = tokenizer(example[0], return_tensors="pt")
    if mode == 'train':
        with tokenizer.as_target_tokenizer():
            for example in examples:
                example[1] = tokenizer(example[1], return_tensors="pt")
    return examples

pat = re.compile(';[A-Z]+?\(.+?\)')
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
    new = [None]
    dicto = defaultdict(list)
    for feat in feats:
        if '-' in feat:
            feat = feat.split('-')
            dicto[feat[0]].append(feat[1])
            if new[-1] != feat[0]:
                new.append(feat[0])
        else:
            new.append(feat)
    new = new[1:]
    for case in dicto:
        new[new.index(case)] = f'{case}({",".join(dicto[case])})'
    return ';'.join(new)

num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lang = argv[1]
size = 'small'
task = argv[2]
data_dir = task
max_len = 15

train_path = os.path.join('/kuacc/users/eacikgoz17/NLP/MRL_shared-task_2022', data_dir, f'{lang}.trn')
test_path = os.path.join('/kuacc/users/eacikgoz17/NLP/MRL_shared-task_2022', data_dir, f'{lang}.dev')
save_path = os.path.join('predictions', 'mT5', f'{lang}_{task}_{size}')

_, train_dataset, morpho_feats = load_dataset(train_path, task)
original_test, test_dataset, _ = load_dataset(test_path, task)

if os.path.isdir(save_path):
    model = MT5ForConditionalGeneration.from_pretrained(save_path).cuda()
    tokenizer = T5Tokenizer.from_pretrained(save_path + '_tok')
    test_dataset = tokenize_dataset(test_dataset, tokenizer, mode='test')
    print('model loaded!')
else:
    model = MT5ForConditionalGeneration.from_pretrained(f"google/mt5-{size}").cuda()
    tokenizer = T5Tokenizer.from_pretrained(f"google/mt5-{size}", sep_token='<sep>')
    num_added_tokens = tokenizer.add_tokens(list(morpho_feats))
    model.resize_token_embeddings(len(tokenizer))
    print('model loaded!')

    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, mode='test')
    print('data loaded!')

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
    tokenizer.save_pretrained(save_path + '_tok')

model.eval()
tot = len(test_dataset)
corrects = 0
out_path = os.path.join('predictions', 'mT5', task, f'{lang}.pred')
out = open(out_path, 'w', encoding='utf8')
for i, (example, target) in enumerate(test_dataset):
    example = example.to(device)
    with torch.no_grad():
        beam_output = model.generate(example.input_ids, max_length=max_len, num_beams=5, early_stopping=True)
        pred = tokenizer.decode(beam_output[0], skip_special_tokens=True)
        if task == 'analysis':
            pred = pred.split()
            feats = ''.join([e for e in pred if e.startswith(';')]).strip(';')
            lemma = ' '.join([e for e in pred if not e.startswith(';')])
            feats = dedist_cases(feats)
            pred = lemma + ' ' + feats
    print(pred, original_test[i][-1])
    out.write('\t'.join(original_test[i][:-1]) + '\t' + pred + '\n')
    if pred == original_test[i][-1]:
        corrects += 1
print(corrects)
