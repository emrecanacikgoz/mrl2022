import torch
import numpy as np
from torch.autograd import Variable

def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    if device == 0:
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, args):
    
    src_mask = (src != 0).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != 0).unsqueeze(-2)
        size = trg.size(-1) # get seq_len for matrix
        np_mask = nopeak_mask(size, args.device)
        if trg.is_cuda:
            np_mask
        trg_mask = trg_mask.to(args.device) & np_mask.to(args.device)
        
    else:
        trg_mask = None
    return src_mask, trg_mask

def plot_curves(task, bmodel, fig, ax, trn_loss_values, val_loss_values, style, ylabel):
    ax.plot(range(len(trn_loss_values)), trn_loss_values, style, label=bmodel+'_trn')
    ax.plot(range(len(val_loss_values)), val_loss_values, style,label=bmodel+'_val')
    if ylabel != 'acc': # hack for clean picture
        leg = ax.legend() #(loc='upper right', bbox_to_anchor=(0.5, 1.35), ncol=3)
        ax.set_title(task,loc='left')
    if ylabel != 'loss':
        ax.set_xlabel('epochs')
    ax.set_ylabel(ylabel)    
def calculate_max_char(parse_data):
    src, tgt = [], []
    for word in parse_data:
        src.append(len(word[0])) 
        tgt.append(len(word[1])) 
    print(f"Max src char: {max(src)} among {len(src)} data")
    print(f"Max tgt char: {max(tgt)} among {len(tgt)} data")