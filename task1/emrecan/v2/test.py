import numpy as np
from utils import create_masks


def test(val_loader, epoch, logger, args):
    epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
    epoch_wrong_predictions = []
    epoch_correct_predictions = []
    for i, (source_lemma, source_tag, source, target) in enumerate(val_loader):

        source_lemma, source_tag, source, target = source_lemma.to(args.device), source_tag.to(args.device), source.to(args.device), target.to(args.device)
        
        # (B, Ty)
        target_input  = target[:, :-1]
        target_output = target[:, 1:]
        
        # src_mask: (B, 1 ,Tx), tgt_mask: (B, Ty, Ty)
        src_mask, trg_mask = create_masks(source, target_input, args)
        src_lemma_mask, _  = create_masks(source_lemma, None, args)
        src_tag_mask, _    = create_masks(source_tag, None, args)

        loss, acc, output = args.model(source, target_input, target_output, epoch, trg_mask, src_lemma_mask, src_tag_mask, src_mask)


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

    if len(wrong_predictions) > 0:
        f1 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_val_wrong_predictions.txt", "w")
        f2 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_val_correct_predictions.txt", "w")
        for i in epoch_wrong_predictions:
            f1.write(i+'\n')
        for i in epoch_correct_predictions:
            f2.write(i+'\n')
        f1.close(); f2.close()

    return nll, ppl, acc