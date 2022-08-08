from openprompt.prompts import SoftVerbalizer, ManualTemplate
from openprompt.plms import load_plm, T5TokenizerWrapper
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader, PromptForClassification
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch

tagno = 1
maxtrnsize = 5000
lang = 'tur'
batchsize = 4
epochs = 10

### DATA
dataset = dict()
raw_dataset = dict()
raw_dataset['train'] = []
raw_dataset['validation'] = []
tag_vocab = {"NA":0}
with open('/home/mugekural/dev/git/competition/task3/task3_muge_v1/analysis/'+lang+'.trn', 'r') as reader:
    for i,line in enumerate(reader):
        data = {}
        split_line = line.strip().split('\t')
        snt = split_line[0]
        lemma= ' '.join(split_line[1].split(' ')[:-1])
        tags = split_line[1].split(' ')[-1]

        data['premise'] = snt
        data['lemma'] = lemma

        if len(tags.split(';')) <tagno+1:
            data['label'] =  tag_vocab['NA']
        else:
            tag = tags.split(';')[tagno] 
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)
            data['label'] =  tag_vocab[tag]
    
        data["idx"] = i
        raw_dataset['train'].append(data)
tag_vocab_id2tags = dict()
for key,val in tag_vocab.items():
    tag_vocab_id2tags[val] = key
with open('/home/mugekural/dev/git/competition/task3/task3_muge_v1/analysis/'+lang+'.dev', 'r') as reader:
    for i,line in enumerate(reader):
        data = {}
        split_line = line.strip().split('\t')
        snt = split_line[0]
        lemma= ' '.join(split_line[1].split(' ')[:-1])
        tags = split_line[1].split(' ')[-1]    
        data['premise'] = snt
        data['lemma'] = lemma

        if len(tags.split(';')) <tagno+1:
            data['label'] =  tag_vocab['NA']
        else:
            tag = tags.split(';')[tagno] 
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)
            data['label'] =  tag_vocab[tag]

        data["idx"] = i
        raw_dataset['validation'].append(data)
for split in ['train', 'validation']:
    dataset[split] = []
    for data in raw_dataset[split][:maxtrnsize]:
        input_example = InputExample(text_a = data['premise'], text_b = data['lemma'], label=int(data['label']), guid=data['idx'])
        dataset[split].append(input_example)


### PRETRAINED_MODEL and PROMPT

# You can load the plm related things provided by openprompt simply by calling:
plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")
# Constructing Template
template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
# To better understand how does the template wrap the example, we visualize one instance.
#wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
#print(wrapped_example)
wrapped_t5tokenizer = WrapperClass(max_seq_length=32, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
# or
wrapped_t5tokenizer= T5TokenizerWrapper(max_seq_length=32, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
# You can see what a tokenized example looks like by
#tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)

model_inputs = {}
for split in ['train', 'validation']:
    model_inputs[split] = []
    for sample in dataset[split]:
        tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
        model_inputs[split].append(tokenized_example)

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=32, decoder_max_length=3,
    batch_size= batchsize,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

# In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:
myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=len(tag_vocab))
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()


## PROMPT TRAINING
# Now the training is standard
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
for epoch in range(epochs):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

### EVALUATE and DUMP RESULTS
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=3,
    batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
with open(lang+'_tag'+str(tagno)+'_preds.txt', 'w') as writer:
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        tokens = []
        tokenizeds = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        for token in tokenizeds:
            if token != '<pad>':
                tokens.append(token)
        writer.write(''.join(tokens)+'\t' + tag_vocab_id2tags[torch.argmax(logits, dim=-1).item()]+'\n')
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    writer.write('acc: %.3f:' % acc)

