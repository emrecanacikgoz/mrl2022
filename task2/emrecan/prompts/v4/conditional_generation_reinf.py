
# # Conditional Generation with Prefix Tuning.
# ref: https://github.com/thunlp/OpenPrompt/blob/main/tutorial/2.1_conditional_generation.py

import argparse
import torch
from openprompt.data_utils import InputExample

parser = argparse.ArgumentParser("")
parser.add_argument("--lang", required=True, type=str)
parser.add_argument("--run", required=True, type=int)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='gpt2')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='sberbank-ai/mGPT')
args = parser.parse_args()
print(args)

dataset = dict()
raw_dataset = dict()
raw_dataset['train'] = []
raw_dataset['validation'] = []
maxtrnsize = 10000

with open('/kuacc/users/eacikgoz17/NLP/shared-task-prompts/task2/v1/reinf/'+args.lang+'.trn', 'r') as reader:
    for i,line in enumerate(reader):
        data = {}
        split_line = line.strip().split('\t')[1:]
        lemma = split_line[0]
        target_snt = split_line[-1]
        target_tags = split_line[-2].split(';')
        full_tag = []
        for tt in target_tags:
            if "(" in tt:
                first, second= tt.split("(")
                second_tags= tt.split("(")[-1][:-1].split(",")
                full_tag.append(first)
                for st in second_tags:
                    full_tag.append(st)
            else: 
                full_tag.append(tt)   

        data['text_a'] =  lemma + ':' +','.join(target_tags) 
        data['tgt_text'] =  target_snt
        data["guid"] = i
        data["text_b"] = ""
        data["meta"] = {}
        data["label"] = None
        raw_dataset['train'].append(data)

with open('/kuacc/users/eacikgoz17/NLP/shared-task-prompts/task2/v1/reinf/'+args.lang+'.dev', 'r') as reader:
    for i,line in enumerate(reader):
        data = {}
        split_line = line.strip().split('\t')[1:]
        lemma = split_line[0]
        target_snt = split_line[-1]
        target_tags = split_line[-2].split(';')
        full_tag = []
        for tt in target_tags:
            if "(" in tt:
                first, second= tt.split("(")
                second_tags= tt.split("(")[-1][:-1].split(",")
                #full_tag.append(abbrevations[first])
                full_tag.append(first)
                for st in second_tags:
                    #full_tag.append(abbrevations[st])
                    full_tag.append(st)
            else:
                #full_tag.append(abbrevations[tt])   
                full_tag.append(tt)      
        data['text_a'] =  lemma + ':' +','.join(full_tag) 
        data['tgt_text'] =  target_snt
        data["guid"] = i
        data["text_b"] = ""
        data["meta"] = {}
        data["label"] = None
        raw_dataset['validation'].append(data)

for split in ['train', 'validation']:
    dataset[split] = []
    for data in raw_dataset[split][:maxtrnsize]:
        input_example = InputExample(text_a = data['text_a'], text_b = data['text_b'], tgt_text =data['tgt_text'], label=None, guid=data['guid'])
        dataset[split].append(input_example)
print(dataset['train'][0])


# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

# Instantiating the PrefixTuning Template !
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
# we can use a plain text as the default setting
# i.e.
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
# is equal to
mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"}')
#mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

# To better understand how does the template wrap the example, we visualize one instance.
# You may observe that the example doesn't end with <|endoftext|> token. Don't worry, adding specific end-of-text token
# is a language-model-specific token. we will add it for you in the TokenizerWrapper once you pass `predict_eos_token=True`
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)
'''from openprompt.plms import T5TokenizerWrapper
wrapped_t5tokenizer= T5TokenizerWrapper(max_seq_length=64, decoder_max_length=32, tokenizer=tokenizer,truncate_method="head")
# You can see what a tokenized example looks like by
tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
print(tokenized_example)
print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))'''

# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=32,
    batch_size=16,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=32,
    batch_size=16,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=32,
    batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration
use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()


from transformers import AdamW
# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]


optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step  = len(train_dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()
    correct = 0
    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
        if output_sentence == inputs['tgt_text']:
            correct+=1
    return generated_sentence, correct


generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": [[628], [198]]
}

# training and generation.
global_step = 0
tot_loss = 0
log_loss = 0
for epoch in range(6):
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        global_step +=1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step %500 ==0:
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss

    generated_sentence, correct = evaluate(prompt_model, test_dataloader)
    with open(args.lang+"_reinflections_epochs"+str(epoch)+"_run"+str(args.run)+".txt",'w') as f:
        for i in generated_sentence:
            f.write(i+"\n")
        f.write('acc: %.3f' % (correct/1000))
        print('acc: %.3f' % (correct/1000))