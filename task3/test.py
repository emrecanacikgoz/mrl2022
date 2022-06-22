import numpy as np
from utils import create_masks


def test(val_loader, epoch, logger, args):
    epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
    epoch_wrong_predictions = []
    epoch_correct_predictions = []
    for i, (surf, feat) in enumerate(val_loader):
        # source: (b, word, txchar)
        # target: (b, word, tychar)
        source, target = surf.to(args.device), feat.to(args.device)

        #sourceBatch, sourcePadded1, sourcePadded2 = source.shape
        we_reshape = source

        # (b*word, txchar)
        source = source.reshape(source.shape[0]*source.shape[1], source.shape[2])

        # (b*word, tychar)
        target = target.reshape(target.shape[0]*target.shape[1], target.shape[2])

        # (b, word, tychar)
        target_input  = target[:, :-1]
        target_output = target[:, 1:]

        # src_mask_we: (b*word, 1, txchar)
        # tgt_mask_we: (b*word, tychar, tychar)
        src_mask_we, trg_mask_we = create_masks(source, target_input, args)
        
        # (b, word)
        _src = we_reshape.sum(-1)
        # (b, 1, word)
        src_mask_ce = create_masks(_src, None,args)[0]
        loss, acc, output = args.model(source, target_input, target_output, we_reshape, epoch, trg_mask_we, None, src_mask_we, src_mask_ce)

        correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = acc
        epoch_loss       += loss.sum().item() #
        epoch_num_tokens += num_tokens
        epoch_acc        += correct_tokens
        epoch_error      += wrong_tokens
        epoch_wrong_predictions   += wrong_predictions
        epoch_correct_predictions += correct_predictions

    nll = epoch_loss / epoch_num_tokens
    ppl = np.exp(epoch_loss / epoch_num_tokens)
    acc = epoch_acc / epoch_num_tokens
    logger.info(f"Epoch: {epoch}/{args.epochs} |  avg_test_loss: {nll:.7f} | perplexity: {ppl:.7f} |  test_accuracy: {acc:.7f}\n")

    f1 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_val_wrong_predictions.txt", "w")
    f2 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_val_correct_predictions.txt", "w")
    for i in epoch_wrong_predictions:
        f1.write(i+'\n')
    for i in epoch_correct_predictions:
        f2.write(i+'\n')
    f1.close(); f2.close()

    return nll, ppl, acc