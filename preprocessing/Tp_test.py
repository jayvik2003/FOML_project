import scipy.io as sio
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob
import os

proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))
import sys
sys.path.append(proj_root_dir)
sys.path.insert(0, '../')
import argparse
from TPrime_transformer.model_transformer import TransformerModel

# CONFIG
TRANS_PATH = '../TPrime_transformer/model_cp'

supported_outmode = ['real', 'complex', 'real_invdim', 'real_ampphase']
PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
CHANNELS = ['None', 'TGn', 'TGax', 'Rayleigh']
SNR = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
np.random.seed(4389) # for reproducibility

def apply_AWGN(snr_dbs, sig):
    rms = np.sqrt(np.mean(np.abs(sig) ** 2))
    sig_W = rms ** 2
    sig_dbW = 10 * np.log10(sig_W / 1)
    noise_dbW = sig_dbW - float(snr_dbs)
    noise_var = 10 ** (noise_dbW / 10)
    noise_std = np.sqrt(noise_var)
    complex_std = noise_std * 1 / np.sqrt(2)
    noise_samples = np.random.normal(0, complex_std, size=sig.shape) + 1j * np.random.normal(0, complex_std, size=sig.shape)
    noisy_sig = sig + noise_samples
    return noisy_sig

def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def validate(model, class_map, seq_len, sli_len, channel):

    correct = np.zeros(len(SNR))
    total_samples = 0
    prev_time = time.time()
    for p in PROTOCOLS:
        path = os.path.join(TEST_DATA_PATH, p) if channel == 'None' else os.path.join(TEST_DATA_PATH, p, channel)
        mat_list = sorted(glob(os.path.join(path, '*.mat'))) if channel == 'None' else sorted(glob(os.path.join(path, '*.npy')))
        for signal_path in mat_list:
            sig = sio.loadmat(signal_path) if channel == 'None' else np.load(signal_path)
            if channel == 'None':
                sig = sig['waveform']
            len_sig = sig.shape[0]
            for i, dBs in enumerate(SNR):
                noisy_sig = apply_AWGN(dBs, sig)
                len_sig = noisy_sig.shape[0]
                obs = np.stack((noisy_sig.real, noisy_sig.imag))
                obs = np.squeeze(obs, axis=2)
                obs = chan2sequence(obs)
                idxs = list(range(seq_len * sli_len * 2, len_sig, seq_len * sli_len * 2))
                obs = np.split(obs, idxs)[:-1]
                for j, seq in enumerate(obs):
                    obs[j] = np.split(seq, seq_len)
                X = np.asarray(obs)
                X = torch.from_numpy(X)
                X = X.to(device)
                y = np.empty(len(idxs))
                y.fill(class_map[p])
                y = torch.from_numpy(y)
                y = y.to(device)
                pred = model(X.float())
                correct[i] += (pred.argmax(1) == y).type(torch.float).sum().item()
                if i == 0:
                    total_samples += len(idxs)
        print("--------", p, "--------")
        print("--- %s seconds for protocol ---" % (time.time() - prev_time))
        prev_time = time.time()
    return correct / total_samples * 100

def generate_dummy_input(channel, seq_len, sli_len):
    p = '802_11ax'
    dBs = 10
    path = os.path.join(TEST_DATA_PATH, p) if channel == 'None' else os.path.join(TEST_DATA_PATH, p, channel)
    mat_list = sorted(glob(os.path.join(path, '*.mat'))) if channel == 'None' else sorted(glob(os.path.join(path, '*.npy')))
    signal_path = mat_list[0]
    sig = sio.loadmat(signal_path) if channel == 'None' else np.load(signal_path)
    if channel == 'None':
        sig = sig['waveform']
    len_sig = sig.shape[0]
    noisy_sig = apply_AWGN(dBs, sig)
    len_sig = noisy_sig.shape[0]
    obs = np.stack((noisy_sig.real, noisy_sig.imag))
    obs = np.squeeze(obs, axis=2)
    obs = chan2sequence(obs)
    idxs = list(range(seq_len * sli_len * 2, len_sig, seq_len * sli_len * 2))
    obs = np.split(obs, idxs)[:-1]
    for j, seq in enumerate(obs):
        obs[j] = np.split(seq, seq_len)
    X = np.asarray(obs)
    X = X[0, :, :]
    X = torch.from_numpy(X)
    X = torch.unsqueeze(X, 0)
    print(X.shape)
    return X

def timing_inference_GPU(device, channel, seq_len, sli_len, model):
    dummy_input = generate_dummy_input(channel, seq_len, sli_len)
    dummy_input = dummy_input.to(device)
    model = model.double()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 30
    timings = np.zeros((repetitions, 1))
    for _ in range(10):
        _ = model(dummy_input.double())
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn

def calculate_avg_time(means, sds):
    weights = 1 / sds ** 2
    weighted_average = np.average(means, weights=weights)
    weighted_std = np.sqrt(np.average((means - weighted_average) ** 2, weights=weights))
    return weighted_average, weighted_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default='3', choices=['1', '2', '3', '4'], help="Decide which models to test, 1 is for models trained for \
                         specific noise and channel conditions, 2 is for models specifically trained for a channel, 3 is for single model for all channel and noise conditions (with SoTA models comparison) and 4 is for inference time analysis")
    parser.add_argument("--normalize", action='store_true', default=False, help="Use a layer norm as a first layer for CNN")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for inference")
    parser.add_argument("--test_path", default='../data/DATASET1_1_TEST', help="Path to the dataset that will be used for testing. DATASET1_1_TEST contains the necessary data to test these models.")
    args, _ = parser.parse_known_args()
    TEST_DATA_PATH = args.test_path
    class_map = dict(zip(PROTOCOLS, range(len(PROTOCOLS))))
    norm_flag = '.norm' if args.normalize else ''
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    if args.experiment == '1':
        for channel in CHANNELS:
            model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=64 * 2, seq_len=24, nlayers=2, use_pos=False)
            model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_10_sm.pt", map_location=device)['model_state_dict'])
            model_sm.eval()
            if args.use_gpu:
                model_sm.cuda()
            y_trans_sm = []
            for test_channel in CHANNELS:
                y_trans_sm.append(validate(model_sm, class_map, seq_len=24, sli_len=64, channel=test_channel))
                print(f'Accuracy values for channel {test_channel} and large architecture trained for {channel} and 10 dBs are: ', y_trans_sm[-1])
            with open(f'test_results_10dBs_{channel}{norm_flag}.txt', 'w') as f:
                f.write(str(y_trans_sm))

    elif args.experiment == '2':
        for channel in CHANNELS:
            model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=128 * 2, seq_len=64, nlayers=2, use_pos=False)
            model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_specific_sm.pt", map_location=device)['model_state_dict'])
            model_sm.eval()
            if args.use_gpu:
                model_sm.cuda()
            y_trans_sm = []
            y_trans_sm.append(validate(model_sm, class_map, seq_len=64, sli_len=128, channel=channel))
            print(f'Accuracy values for channel {channel} and large architecture are: ', y_trans_sm[-1])
            with open(f'test_results_specific_{channel}{norm_flag}.txt', 'w') as f:
                f.write(str(y_trans_sm))

    elif args.experiment == '3':
        for channel in CHANNELS:
            model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=128 * 2, seq_len=64, nlayers=2, use_pos=False)
            model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_range_sm.pt", map_location=device)['model_state_dict'])
            model_sm.eval()
            if args.use_gpu:
                model_sm.cuda()
            y_trans_sm = []
            y_trans_sm.append(validate(model_sm, class_map, seq_len=64, sli_len=128, channel=channel))
            print(f'Accuracy values for channel {channel} and large architecture are: ', y_trans_sm[-1])
            with open(f'test_results_general_{channel}{norm_flag}.txt', 'w') as f:
                f.write(str(y_trans_sm))

    elif args.experiment == '4':
        mean_syn_trans_sm = []
        std_syn_trans_sm = []
        for channel in CHANNELS:
            model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=128 * 2, seq_len=64, nlayers=2, use_pos=False)
            model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_range_sm.pt", map_location=device)['model_state_dict'])
            model_sm.eval()
            if args.use_gpu:
                model_sm.cuda()
            mean_syn_sm, std_syn_sm = timing_inference_GPU(device, channel, 64, 128, model_sm)
            mean_syn_trans_sm.append(mean_syn_sm)
            std_syn_trans_sm.append(std_syn_sm)
            print(f'Timing for channel {channel} with large transformer architecture: Mean {mean_syn_sm}, Std {std_syn_sm}')
        avg_mean_sm, avg_std_sm = calculate_avg_time(np.array(mean_syn_trans_sm), np.array(std_syn_trans_sm))
        print(f'Avg mean and std for transformer large is: {avg_mean_sm} and {avg_std_sm}')
        with open(f'inference_timing_transformer_large{norm_flag}.txt', 'w') as f:
            f.write(str(mean_syn_trans_sm))
            f.write('\n')
            f.write(str(std_syn_trans_sm))
            f.write('\n')
            f.write(str(avg_mean_sm))
            f.write('\n')
            f.write(str(avg_std_sm))