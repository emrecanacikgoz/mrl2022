import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import Encoder, Decoder

class Morse(nn.Module):
    def __init__(self, input_vocab, output_vocab, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(Morse, self).__init__()

        self.feat = input_vocab
        self.surf = output_vocab

        self.encoder    = Encoder(len(input_vocab),  embed_dim, num_heads, dropout_rate)
        self.decoder    = Decoder(len(output_vocab), embed_dim, num_heads, dropout_rate)
        self.linear     = nn.Linear(embed_dim, len(output_vocab))


    def forward(self, input, target_in, target_out, epoch, trg_mask=None, src_mask=None):

        # (B, Tx) 
        encoder_output = self.encoder(input, src_mask)
        # (B, Tx, embed)
        decoder = self.decoder(target_in, encoder_output, trg_mask, src_mask)
        # (B, Ty, embed)
        output = self.linear(decoder)
        # (B, Ty, vocab_size)
        _output = output.view(-1, output.size(-1))
        # (Ty, vocab)
        _target = target_out.contiguous().view(-1)
        # (B*Ty)
        loss = F.cross_entropy(_output, _target, ignore_index=0, reduction='none')
        # (B*Ty)
        return loss, self.accuracy(output, target_out, epoch), output
    
    
    def accuracy(self, outputs, targets, epoch):
        # output_logits: (B, T, vocab_size), targets: (B, T)
        surf_vocab = self.surf
        feat_vocab = self.feat

        B = targets.size(0)
        softmax = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(softmax(outputs),2)
        
        correct_tokens = (pred_tokens == targets) * (targets!=0) # padid=0, this line should be more generic
        wrong_tokens   = (pred_tokens != targets) * (targets!=0)

        num_correct = correct_tokens.sum().item() 
        num_wrong   = wrong_tokens.sum().item() 
        num_total   = num_correct + num_wrong # also equal to (targets!=0).sum()

        # Log predictions into file
        correct_predictions = []; wrong_predictions = []
        if (epoch % 5) == 0:
            for i in range(B):
                target  = ''.join([surf_vocab[seq.item()] for seq in targets[i]])
                pred  = ''.join([surf_vocab[seq.item()] for seq in pred_tokens[i]])
                if '</s>' not in target: # ignore dummy inputs with full of pads
                    continue
                target = target[:target.index('</s>')+4] #take until eos
                pred = pred[:len(target)]
                if target != pred:
                    wrong_predictions.append('target: %s pred: %s' % (target, pred))
                else:
                    correct_predictions.append('target: %s pred: %s' % (target, pred))

        return  num_correct, num_total, num_wrong, wrong_predictions, correct_predictions