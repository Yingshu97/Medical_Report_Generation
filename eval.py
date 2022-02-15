import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from IUdataloaders import R2DataLoader
from datasets import *
from utils import *
import torch.nn.functional as F
from Tokenizer import Tokenizer
from tqdm import tqdm
import argparse
from metrics import compute_scores

# Parameters
data_name = 'IU_Xray'   # base name shared by data files
checkpoint = './BEST_checkpoint_IU_Xray.pth.tar'  # model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=80, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    # parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples for a batch')  # test in 1

    args = parser.parse_args()
    return args


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    args = parse_agrs()
    tokenizer = Tokenizer(args)
    vocab_size = tokenizer.get_vocab_size()

    # DataLoader
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (images_id, images, reports_ids, reports_masks, caplens) in enumerate(
            tqdm(test_dataloader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        images = images.to(device)
        reports_masks = reports_masks.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        encoder_out = encoder(images)  # [batch_size, 98, 2048]

        # Encode
        enc_image_size = int((encoder_out.size(1)/2) ** 0.5)
        encoder_dim = encoder_out.size(2)

        # Flatten encoding
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[tokenizer.get_id_by_token('SOS')]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != tokenizer.get_id_by_token('EOS')]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds].long()]
            c = c[prev_word_inds[incomplete_inds].long()]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 120:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = reports_ids.tolist()
        img_caption = list(filter(lambda c: c not in {tokenizer.get_id_by_token('pad'), 
                                                            tokenizer.get_id_by_token('SOS'), 
                                                            tokenizer.get_id_by_token('EOS')},img_caps[0])) # remove <start> and pads

        words = []
        for id in img_caption:
            words.append(tokenizer.get_token_by_id(id))
        references.append(img_caption)

        # Hypotheses
        img_caption_h = list(filter(lambda c: c not in {tokenizer.get_id_by_token('pad'), 
                                                            tokenizer.get_id_by_token('SOS'), 
                                                            tokenizer.get_id_by_token('EOS')},seq)) # remove <start> and pads
        words = []
        for id in img_caption_h:
            words.append(tokenizer.get_token_by_id(id))                                                    
        hypotheses.append(img_caption_h)
        
        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    ref = tokenizer.decode_batch(references)
    hy = tokenizer.decode_batch(hypotheses)

    # ref ['xxxxx','xxxxxx','sdasdfasdf']
    # pred: ['12314','sadfasdf','asdasda']
    scores = compute_scores({i: [gt] for i, gt in enumerate(ref)},
                                        {i: [re] for i, re in enumerate(hy)})

    return scores


if __name__ == '__main__':
    beam_size = 3
    print(f'\nThe scores @ beam size of {beam_size}: {evaluate(beam_size)}')
