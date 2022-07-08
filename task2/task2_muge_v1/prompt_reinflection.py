import argparse
import torch
from openprompt.data_utils import InputExample
from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor
from openprompt.plms import load_plm
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptDataLoader, PromptForGeneration
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
args = parser.parse_args()
print(args)

dataset = dict()
raw_dataset = dict()
raw_dataset['train'] = []
raw_dataset['validation'] = []
maxtrnsize = 10000
epochs = 10
batchsize = 4

plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")
template_text = '{"placeholder":"text_a"} {"placeholder":"text_b"} {"special": "<eos>"} {"mask"}'
mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=template_text, using_decoder_past_key_values=False)


### DATA
with open('/home/mugekural/dev/git/competition/task2/task2_muge_v1/reinf/eng.trn', 'r') as reader:
    for i,line in enumerate(reader):
        data = {}
        split_line = line.strip().split('\t')
        source_tag, source_snt, target_tag, target_snt = split_line
        snt = split_line[0]
        data['text_a']   =  source_snt
        data['text_b']   =  target_tag
        data['tgt_text'] =  target_snt
        data["guid"] = i
        data["meta"] = {}
        data["label"] = None
        raw_dataset['train'].append(data)

with open('/home/mugekural/dev/git/competition/task2/task2_muge_v1/reinf/eng.dev', 'r') as reader:
    for i,line in enumerate(reader):
        data = {}
        split_line = line.strip().split('\t')
        source_tag, source_snt, target_tag, target_snt = split_line
        snt = split_line[0]
        data['text_a']   =  source_snt
        data['text_b']   =  target_tag
        data['tgt_text'] =  target_snt
        data["guid"] = i
        data["meta"] = {}
        data["label"] = None
        raw_dataset['validation'].append(data)

for split in ['train', 'validation']:
    dataset[split] = []
    for data in raw_dataset[split][:maxtrnsize]:
        input_example = InputExample(text_a = data['text_a'], text_b = data['text_b'], tgt_text = data['tgt_text'], label=None, guid=data['guid'])
        dataset[split].append(input_example)

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])


train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=64,
    batch_size=batchsize, shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=64,
    batch_size=batchsize, shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=64, decoder_max_length=64,
    batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# load the pipeline model PromptForGeneration.
use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()


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
tot_step  = len(train_dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

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
    "min_length": 1,
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
for epoch in range(epochs):
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
        if global_step %50 ==0:
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss

generated_sentence, correct = evaluate(prompt_model, test_dataloader)

with open(f"reinflections.txt",'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")
    f.write('acc: %.3f' % (correct/1000))
    print('acc: %.3f' % (correct/1000))


