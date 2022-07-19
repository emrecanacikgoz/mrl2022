import random, torch, time, logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from test import test



def train(train_loader, val_loader, logger, args):
    scheduler = ReduceLROnPlateau(args.optimizer, mode='min', factor=0.8, patience=3, verbose=True)

    best_loss = 1e4
    start_time = time.time()
    trn_loss_values, trn_acc_values= [], []
    val_loss_values, val_acc_values= [], []
    for epoch in range(args.epochs):
        args.model.train()
        epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0;
        epoch_wrong_predictions, epoch_correct_predictions = [], []
        for i, (source, target) in enumerate(train_loader):
            args.model.zero_grad()

            # source: (B, Tx), target: (B, Ty)
            source, target = source.to(args.device), target.to(args.device)

            # (B, Ty)
            target_input  = target[:, :-1]
            target_output = target[:, 1:]
            
            # src_mask: (B, 1 ,Tx), tgt_mask: (B, Ty, Ty)
            src_mask, trg_mask = create_masks(source, target_input, args)

            loss, acc, output = args.model(source, target_input, target_output, epoch, trg_mask, src_mask)

            batch_loss = loss.mean() # optimize for mean loss per token
            batch_loss.backward()
            args.optimizer.step()

            correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions, _ = acc
            epoch_loss       += loss.sum().item()
            epoch_num_tokens += num_tokens
            epoch_acc        += correct_tokens
            epoch_error      += wrong_tokens
            epoch_wrong_predictions   += wrong_predictions
            epoch_correct_predictions += correct_predictions
            #print(f"\nBatch: {i+1}/{len(train_loader)} Loss: {batch_loss:.5f} Acc: {correct_tokens/num_tokens:.5f}\n")

        nll_train = epoch_loss / epoch_num_tokens #len(train_loader)
        ppl_train = np.exp(epoch_loss / epoch_num_tokens)
        acc_train = epoch_acc / epoch_num_tokens
        trn_loss_values.append(nll_train)
        trn_acc_values.append(acc_train)
        #print(f"Epoch: {epoch}/{args.epochs} | avg_train_loss: {nll_train:.7f} | perplexity: {ppl_train:.7f} | train_accuracy: {acc_train:.7f}")
        logger.info(f"Epoch: {epoch}/{args.epochs} | avg_train_loss: {nll_train:.7f} | perplexity: {ppl_train:.7f} | train_accuracy: {acc_train:.7f}")

        # File Operations
        if len(wrong_predictions) > 0:
            f1 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_trn_wrong_predictions.txt", "w")
            f2 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_trn_correct_predictions.txt", "w")
            for i in epoch_wrong_predictions:
                f1.write(i+'\n')
            for i in epoch_correct_predictions:
                f2.write(i+'\n')
            f1.close(); f2.close()

        # Validation
        args.model.eval()
        with torch.no_grad():
            nll_test, ppl_test, acc_test = test(val_loader, epoch, logger, args)
            loss = nll_test
        val_loss_values.append(nll_test)
        val_acc_values.append(acc_test)
        scheduler.step(nll_test)

        # Savings
        if loss < best_loss:
            logger.info('Update best val loss\n')
            best_loss = loss
            best_ppl = ppl_test
            torch.save(args.model.state_dict(), args.save_path)

        logging.info("\n")

    end_time = time.time()
    training_time = (abs(end_time - start_time))
    logger.info(f"\n\n---Final Results---")
    #print(f"Epochs: {args.epochs}, Batch Size: {args.batchsize}, lr: {args.lr}, train_loss: {nll_train:.4f}, val_loss: {nll_test:.4f}")
    logger.info(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, lr: {args.lr}, train_loss: {nll_train:.4f}")
    logger.info(f"Training Time: {training_time}\n")
    plot_curves(args.task, args.mname, args.fig, args.axs, trn_loss_values, val_loss_values, args.plt_style, 'loss')
