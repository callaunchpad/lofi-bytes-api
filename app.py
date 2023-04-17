import torch
import torch.nn as nn
import os
import random
import pretty_midi
import processor

from flask import Flask, jsonify, request

from processor import encode_midi, decode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.device import cpu_device

SEQUENCE_START = 0

OUTPUT_PATH = "../output"

MODEL_WEIGHTS = '../model_lofi/results/best_loss_weights.pickle'

RPR = ""

TARGET_SEQ_LENGTH = 1023

NUM_PRIME = 65

MAX_SEQUENCE = 2048

N_LAYERS = 6

NUM_HEADS = 8

D_MODEL = 512

DIM_FEEDFORWARD = 1024
    
# main
def main(primer_midi):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    '''if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")'''

    #os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Grabbing dataset if needed
    #_, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    # Can be None, an integer index to dataset, or a file path
    '''if(args.primer_file is None):
        f = str(random.randrange(len(dataset)))
    else:
        f = args.primer_file'''

    '''

    if(f.isdigit()):
        idx = int(f)
        primer, _  = dataset[idx]
        primer = primer.to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

    '''

    
    raw_mid = encode_midi(primer_midi)
    if(len(raw_mid) == 0):
        print("Error: No midi messages in primer file:", f)
        return

    primer, _  = process_midi(raw_mid, NUM_PRIME, random_seq=False)
    primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

    print("Using primer file:", f)

    model = MusicTransformer(n_layers=N_LAYERS, num_heads=NUM_HEADS,
                d_model=D_MODEL, dim_feedforward=DIM_FEEDFORWARD,
                max_sequence=MAX_SEQUENCE, rpr=RPR).to(get_device())

    model.load_state_dict(torch.load(MODEL_WEIGHTS))

    # Saving primer first
    f_path = os.path.join(OUTPUT_PATH, "primer.mid")
    decode_midi(primer[:NUM_PRIME].cpu().numpy(), file_path=f_path)



    # GENERATION 
    # TODO: do not store the file, return processed midi back to caller
    model.eval()
    with torch.set_grad_enabled(False):
        if(args.beam > 0):
            print("BEAM:", args.beam)
            beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)

            f_path = os.path.join(args.output_dir, "beam.mid")
            decode_midi(beam_seq[0].cpu().numpy(), file_path=f_path)
        else:
            print("RAND DIST")
            rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)

            f_path = os.path.join(args.output_dir, "rand.mid")
            decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)


# process_midi
def process_midi(raw_mid, max_seq, random_seq):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    tgt = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    if(raw_len < full_seq):
        x[:raw_len]         = raw_mid
        tgt[:raw_len-1]     = raw_mid[1:]
        tgt[raw_len]        = TOKEN_END
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]


    # print("x:",x)
    # print("tgt:",tgt)

    return x, tgt


if __name__ == "__main__":
    main()
