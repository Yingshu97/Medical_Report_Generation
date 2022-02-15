import time
import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from Tokenizer import Tokenizer
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from IUdataloaders import R2DataLoader
from metrics import compute_scores

# Data parameters
data_name = 'IU_Xray'  # base name shared by data files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 8
# workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
best_loss = 100.00
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = 'BEST_checkpoint_IU_Xray.pth.tar'

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=80, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')
    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of word embeddings')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers')
    parser.add_argument('--dropout', type=int, default=0.5, help='dimension of attention linear layers')

    args = parser.parse_args()
    return args

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, best_loss
    args = parse_agrs()

    # create tokenizer
    tokenizer = Tokenizer(args)
    total_vocab_size = tokenizer.get_vocab_size()

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    

    if checkpoint is None:
        decoder = DecoderWithAttention(args, vocab_size=total_vocab_size)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)    

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        # train(train_loader=train_dataloader,
        #       encoder=encoder,
        #       decoder=decoder,
        #       criterion=criterion,
        #       encoder_optimizer=encoder_optimizer,
        #       decoder_optimizer=decoder_optimizer,
        #       epoch=epoch)

        # One epoch's validation
        recent_bleu4, recent_loss = validate(val_loader=val_dataloader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                tokenizer=tokenizer)

        # Check if there was an improvement
        is_best = recent_loss < best_loss
        # is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        best_loss = min(recent_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top2accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (images_id, images, reports_ids, reports_masks, caplens) in enumerate(train_loader):
        images = images.to(device)
        reports_ids = reports_ids.to(device)
        reports_masks = reports_masks.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(images)  # [batch_size, 98, 2048]
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, reports_ids, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        scores=scores.data
        targets=targets.data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top2 = accuracy(scores, targets, 2)
        losses.update(loss.item(), sum(decode_lengths))
        top2accs.update(top2, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-2 Accuracy {top2.val:.3f} ({top2.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top2=top2accs))

def validate(val_loader, encoder, decoder, criterion, tokenizer):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top2accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (images_id, images, reports_ids, reports_masks, caplens) in enumerate(val_loader):
            images = images.to(device)
            reports_ids = reports_ids.to(device)
            reports_masks = reports_masks.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(images)  # [batch_size, 98, 2048]
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, reports_ids, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores= pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets= pack_padded_sequence(targets, decode_lengths, batch_first=True)
            scores=scores.data
            targets=targets.data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top2 = accuracy(scores, targets, 2)
            top2accs.update(top2, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-2 Accuracy {top2.val:.3f} ({top2.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top2=top2accs))
            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            img_caps = caps_sorted.tolist()
            words = []
            for id in img_caps[0]:
                words.append(tokenizer.get_token_by_id(id))
            
            img_captions = []
            for i in range(len(img_caps)):
                # bleu need a list of reference so we need to add another layer
                img_caption = list(filter(lambda c: c not in {tokenizer.get_id_by_token('pad'), 
                                                            tokenizer.get_id_by_token('SOS'), 
                                                            tokenizer.get_id_by_token('EOS')},img_caps[i])) # remove <start> and pads
                img_captions.append(img_caption)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            pred_for_eval = list()

            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:int(decode_lengths[j])])  # remove pads

                if 2 in preds[j]:
                    pred_for_eval.append(preds[j][:int(preds[j].index(2))])  # stop at EOS
                else:
                    pred_for_eval.append(preds[j][:int(decode_lengths[j])])
            preds = temp_preds


            words = []
            for id in pred_for_eval[0]:
                words.append(tokenizer.get_token_by_id(id))

            p = tokenizer.decode_batch(pred_for_eval)
            c = tokenizer.decode_batch(img_captions)
            hypotheses.extend(p)
            references.extend(c)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        scores = compute_scores({i: [gt] for i, gt in enumerate(references)},
                                        {i: [re] for i, re in enumerate(hypotheses)})

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-2 ACCURACY - {top2.avg:.3f}, COCO_eval - {bleu}\n'.format(
                loss=losses,
                top2=top2accs,
                bleu=scores))
    return scores['BLEU_4'], losses.avg


if __name__ == '__main__':
    main()
