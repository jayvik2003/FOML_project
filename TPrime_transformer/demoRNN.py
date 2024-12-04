import scipy.io as sio
import time
import numpy as np
import torch
import os
from glob import glob
import argparse

# Define the project root directory
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))

# Update system path for imports
import sys
sys.path.append(proj_root_dir)
sys.path.insert(0, '../')

from model_rnn import RNNMLPClassifier

# Define paths, protocols, and parameters
TRANS_PATH = './model_cp'
PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
CHANNELS = ['None']
pp = ['802_11ax']  # Protocols to process
SNR = [0]          # Signal-to-noise ratios
SN = [0]           # Signal indices to process
np.random.seed(4389)

def chan2sequence(obs):
    """Convert channel data to a sequence."""
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def process_single_packet(model, class_map, seq_len, Slice_length, channel, protocol):
    """Process a single packet using the provided model."""
    path = os.path.join(TEST_DATA_PATH, protocol)
    mat_list = sorted(glob(os.path.join(path, '*.mat')))
    if not mat_list:
        print(f"No .mat files found in {path}. Skipping protocol {protocol}.")
        return
    for i in SN:
        if i >= len(mat_list):
            print(f"Index {i} out of range for protocol {protocol}. Skipping.")
            continue
        signal_path = mat_list[i]
        print(f"Signal path: {signal_path}")
        sig = sio.loadmat(signal_path)
        if channel == 'None':
            sig = sig['waveform']
            len_sig = sig.shape[0]
            for dBs in SNR:
                noisy_sig = sig  # No noise added as apply_AWGN is removed
                len_sig = noisy_sig.shape[0]
                obs = np.stack((noisy_sig.real, noisy_sig.imag))
                obs = np.squeeze(obs, axis=2)
                obs = chan2sequence(obs)
                idxs = list(range(seq_len * Slice_length * 2, len_sig, seq_len * Slice_length * 2))
                obs = np.split(obs, idxs)[:-1]
                for j, seq in enumerate(obs):
                    obs[j] = np.split(seq, seq_len)
                X = np.asarray(obs)
                X = torch.from_numpy(X).to(device)
                y = np.empty(len(idxs))
                y.fill(class_map[protocol])
                y = torch.from_numpy(y).to(device)
                start_time = time.time()
                pred = model(X.float())
                end_time = time.time()
                print(f"Processing time for protocol {protocol}, channel {channel}: {end_time - start_time:.4f} seconds")
                print(f"Accuracy: {(pred.argmax(1) == y).type(torch.float).sum().item() / len(y) * 100:.2f}%")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    
    # Test data path
    TEST_DATA_PATH = 'DATASET1_1_TEST'
    class_map = dict(zip(PROTOCOLS, range(len(PROTOCOLS))))
    device = torch.device("cuda")

    # Slice configuration
    k_list = ['lg']
    for k in k_list:
        if k == 'lg':
            Slice_length = 128
            input_size = 2 * Slice_length  # Input size for the RNN
            seq_len = 64
            hidden_size = 64  # Adjust based on your RNN configuration
        elif k == 'sm':
            Slice_length = 64
            input_size = 2 * Slice_length
            seq_len = 24
            hidden_size = 64  # Adjust based on your RNN configuration

        # Load RNN model
        model = RNNMLPClassifier(
            input_size=input_size,  # Account for interleaved IQ data
            hidden_size=hidden_size,
            seq_len=seq_len,
            num_layers=2,  # Match the RNN depth
            dropout=0.1,   # Match the dropout used
            num_classes=len(PROTOCOLS)
        )

        # Load model weights
        kk_list = ['lg','sm']
        for k in k_list:
            if k == 'lg':
                model.load_state_dict(torch.load(f"{TRANS_PATH}/modelNone_range_lg.pt", map_location=device)['model_state_dict'])
            else:
                model.load_state_dict(torch.load(f"{TRANS_PATH}/modelNone_range_sm.pt", map_location=device)['model_state_dict'])

            model.eval()
            model.cuda()

            # Process packets for each protocol and channel
            for protocol in pp:
                for channel in CHANNELS:
                    print(f"\nProcessing single packet for protocol {protocol}, channel {channel}")
                    process_single_packet(model, class_map, seq_len=seq_len, Slice_length=Slice_length, channel=channel, protocol=protocol)

            print(f'\nThis was completed for configuration {k}.\n')
