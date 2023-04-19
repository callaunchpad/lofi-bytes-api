import torch
import torch.nn as nn
import os
import random
import pretty_midi
import processor

from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, flash, redirect, url_for

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

MODEL_WEIGHTS = '../best_loss_weights.pickle'

RPR = ""

TARGET_SEQ_LENGTH = 1023

NUM_PRIME = 65

MAX_SEQUENCE = 2048

N_LAYERS = 6

NUM_HEADS = 8

D_MODEL = 512

DIM_FEEDFORWARD = 1024

BEAM = 0

FORCE_CPU = True

ALLOWED_EXTENSIONS = {'mid'}

app = Flask(__name__)

app.secret_key = 'super secret'

app.config['UPLOAD_FOLDER'] = '/Users/ericliu/Launchpad/lofi-bytes-api/uploaded_midis'

generated_midi = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        
        if 'file' not in request.files:
            flash('No file part')
            return redirect("https://www.google.com/")
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            #this is where we call generate on the midi and use model to create the output midi that FRONTEND should play 
            #for the user
            
            generated_midi = generate(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #ML Model Music Generation WORKS!!!
            #TODO: ASK ALICIA IF GENERATED MUSIC IS OK, SEEMS A LITTLE BAD BECAUSE THE
            # GENERATED MUSIC SOUNDS BAD --> COULD BE CAUSED BY SETTING strict=false in load_state_dict
            #TODO: PASS THE generated_midi to frontend to PLAY the audio

            return redirect("http://localhost:5173/home/")
            
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
    
# main
def generate(primer_midi):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    if(FORCE_CPU):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

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
        return

    primer, _  = process_midi(raw_mid, NUM_PRIME, random_seq=False)
    primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=cpu_device())



    model = MusicTransformer(n_layers=N_LAYERS, num_heads=NUM_HEADS,
                d_model=D_MODEL, dim_feedforward=DIM_FEEDFORWARD,
                max_sequence=MAX_SEQUENCE, rpr=RPR).to(cpu_device())
    
    #model.load_state_dict(torch.load(MODEL_WEIGHTS))

    state_dict = torch.load(MODEL_WEIGHTS, map_location=cpu_device())

    #torch.save(state_dict)

    #print(state_dict)
    #print(model.state_dict().keys())

    
    #transformer.encoder.layers.0.self_attn.Er is not being used in state_dict!!??
    #no self attention error?
    model.load_state_dict(state_dict, strict=False) #does strict=False fuck up the model?

    # Saving primer first
    f_path = os.path.join(OUTPUT_PATH, "primer.mid")

    # saves a pretty_midi at file_path
    decode_midi(primer[:NUM_PRIME].cpu().numpy(), file_path=f_path) 



    # GENERATION 
    model.eval()
    with torch.set_grad_enabled(False):
  
        
        #model.generate() returns a MIDI stored as an ARRAY given a primer
        beam_seq = model.generate(primer[:NUM_PRIME], TARGET_SEQ_LENGTH, beam=BEAM)

        #save beam_seq in a file for testing purposes
        f_path = os.path.join(OUTPUT_PATH, "pretty_midi.mid")

        #decode_midi() returns an actual MIDI of class pretty_midi.PrettyMIDI
        decoded_midi = decode_midi(beam_seq[0].cpu().numpy(), file_path=f_path)

        #THIS SHOULD BE EITHER decoded_midi OR beam_seq
        #TODO: decoded_midi is actual pretty_midi MIDI file, beam_seq is just an array representing a MIDI
        #decoded_midi stores more information about instruments and stuff
        #returning decoded_midi seems more legit
        return decoded_midi
        
        '''else:
            print("RAND DIST")
            rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)

            f_path = os.path.join(args.output_dir, "rand.mid")
            decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)'''


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


if __name__ == '__main__':
    app.run()
