import torch
import torch.nn as nn
import torch.nn.functional as F
from model.morse_layers import WordEncoder, ContextEncoder, WordDecoder

class Morse(nn.Module):
    def __init__(self, input_vocab, output_vocab, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(Morse, self).__init__()

        self.surf = input_vocab
        self.feat = output_vocab

        self.word_encoder    = WordEncoder(len(input_vocab),  embed_dim, num_heads, dropout_rate)
        self.context_encoder = ContextEncoder(embed_dim, num_heads, dropout_rate)
        self.word_decoder    = WordDecoder(len(output_vocab), embed_dim*2, num_heads, dropout_rate)
        self.linear          = nn.Linear(embed_dim*2, len(output_vocab))


    def forward(self, input, target_in, target_out, we_reshape, epoch, trg_mask_we=None, trg_mask_ce=None, src_mask_we=None, src_mask_ce=None):
        # we_reshape:  (b, word, txchar)
        # input:       (b*word, txchar) 
        # src_mask_we: (b*word, 1, txchar)
        # src_mask_ce: (b, 1, word)
        batchsize, word, txchar = we_reshape.size()

        # we_output:        (b*word, txchar, wordenc_dim)
        # we_output_avg:    (b, word, wordenc_dim)
        we_output, we_output_avg = self.word_encoder(input, we_reshape, src_mask_we)
        # (b, word, contenc_dim)
        ce_output = self.context_encoder(we_output_avg, src_mask_ce)
        # (b*word, 1, contenc_dim)
        ce_output = ce_output.reshape(ce_output.shape[0]*ce_output.shape[1], ce_output.shape[2]).unsqueeze(-2)
        # (b*word, txchar, contenc_dim)
        ce_output = ce_output.expand(batchsize*word, txchar, -1)
        
        # (b*word, txchar, wordenc_dim + contenc_dim)
        encoder_output = torch.cat((we_output,ce_output),dim=2)
        
        # (b*word, tychar, worddec_dim)
        we_decoder  = self.word_decoder(target_in, encoder_output, trg_mask_we, src_mask_we)
        
        # (b*word, tychar, vocab_size)
        output      = self.linear(we_decoder)
        # (b*word*tychar, vocab_size)
        _output = output.view(-1, output.size(-1))
        # (b*word*tychar)
        _target = target_out.contiguous().view(-1)
        # (b*word*tychar)
        loss    = F.cross_entropy(_output, _target, ignore_index=0, reduction='none')
        return loss, self.accuracy(output, target_out, epoch), output
    
    def accuracy(self, output_logits, targets, epoch):
        # output_logits: (B, T, vocab_size), targets: (B,T)
        surf_vocab = self.surf
        feat_vocab = self.feat

        B, T = targets.size()
        softmax = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(softmax(output_logits),2)
        
        correct_tokens = (pred_tokens == targets) * (targets!=0) # padid=0, this line should be more generic
        wrong_tokens   = (pred_tokens != targets) * (targets!=0)

        num_correct = correct_tokens.sum().item() 
        num_wrong   = wrong_tokens.sum().item() 
        num_total   = num_correct + num_wrong # also equal to (targets!=0).sum()

        # Log predictions into file
        correct_predictions = []; wrong_predictions = []
        if epoch % 25 == 0:
            for i in range(B):
                target  = ''.join([feat_vocab[seq.item()] for seq in targets[i]])
                pred  = ''.join([feat_vocab[seq.item()] for seq in pred_tokens[i]])
                if '</s>' not in target: # ignore dummy inputs with full of pads
                    continue
                target = target[:target.index('</s>')+4] #take until eos
                pred = pred[:len(target)]
                if target != pred:
                    wrong_predictions.append('target: %s pred: %s' % (target, pred))
                else:
                    correct_predictions.append('target: %s pred: %s' % (target, pred))

        return  num_correct, num_total, num_wrong, wrong_predictions, correct_predictions