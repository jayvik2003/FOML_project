import scipy.io as sio
import time
import numpy as np
import torch
import os
from glob import glob
import argparse

proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))

import sys
sys.path.append(proj_root_dir)
sys.path.insert(0, '../')

from TPrime_transformer.model_rnn import RNNMLPClassifier

TRANS_PATH = '../TPrime_transformer/model_cp'
PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
CHANNELS = ['None']
pp = ['802_11ax']
SNR = [0]
SN = [0]
np.random.seed(4389)

def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def process_single_packet(model, class_map, seq_len, hidden_size, channel, protocol):
    path = os.path.join(TEST_DATA_PATH, protocol)
    mat_list = sorted(glob(os.path.join(path, '*.mat')))
    for i in SN:
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
                idxs = list(range(seq_len * hidden_size * 2, len_sig, seq_len * hidden_size * 2))
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
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    TEST_DATA_PATH = 'DATASET1_1_TEST'
    class_map = dict(zip(PROTOCOLS, range(len(PROTOCOLS))))
    device = torch.device("cuda")

    k_list = ['lg']
    for k in k_list:
        if k == 'lg':
            input_size = 128  # Input size for the RNN
            seq_len = 64
            hidden_size = 256  # Adjust based on your RNN configuration
        elif k == 'sm':
            input_size = 64
            seq_len = 24
            hidden_size = 128  # Adjust based on your RNN configuration

        model = RNNMLPClassifier(
            input_size=input_size * 2,  # Account for interleaved IQ data
            hidden_size=hidden_size,
            seq_len=seq_len,
            num_layers=2,  # Match the RNN depth
            dropout=0.1,   # Match the dropout used
            num_classes=len(PROTOCOLS)
        )

        if k == 'lg':
            model.load_state_dict(torch.load(f"{TRANS_PATH}/modelNone_range_lg.pt", map_location=device)['model_state_dict'])
        else:
            model.load_state_dict(torch.load(f"{TRANS_PATH}/modelNone_range_sm.pt", map_location=device)['model_state_dict'])

        model.eval()
        model.cuda()

        for protocol in pp:
            for channel in CHANNELS:
                print(f"\nProcessing single packet for protocol {protocol}, channel {channel}")
                process_single_packet(model, class_map, seq_len=seq_len, hidden_size=hidden_size, channel=channel, protocol=protocol)

        print(f'\nThis was for completed {k} \n ')
