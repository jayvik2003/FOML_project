import scipy.io as sio
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from torchviz import make_dot

proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))

import sys

sys.path.append(proj_root_dir)
sys.path.insert(0, '../')

import argparse
from TPrime_transformer.demo_model import TransformerModel

# CONFIG
TRANS_PATH = '../TPrime_transformer/model_cp'
supported_outmode = ['real', 'complex', 'real_invdim', 'real_ampphase','d']
PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
CHANNELS = ['None']#, 'TGn', 'TGax', 'Rayleigh']
pp = ['802_11ax']#, '802_11b_upsampled', '802_11n', '802_11g']
SNR =[-10]
SN = [0]  # Only one SNR level for simplicity
np.random.seed(4389)  # for reproducibility

def apply_AWGN(snr_dbs, sig):
    rms = np.sqrt(np.mean(np.abs(sig) ** 2))
    sig_W = rms ** 2
    sig_dbW = 10 * np.log10(sig_W / 1)
    noise_dbW = sig_dbW - float(snr_dbs)
    noise_var = 10 ** (noise_dbW / 10)
    noise_std = np.sqrt(noise_var)
    complex_std = noise_std * 1 / np.sqrt(2)
    noise_samples = np.random.normal(0, complex_std, size=sig.shape) + 1j * np.random.normal(0, complex_std, size=sig.shape)
    noisy_sig = sig + 0*noise_samples
    
    sio.savemat('noisy_signal.mat',{'noisy':noisy_sig})

    return noisy_sig

def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def plot_heatmap(data, title):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, cmap='viridis')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Feature Dimension')
    plt.show()

def process_single_packet(model, class_map, seq_len, sli_len, channel, protocol):
    path = os.path.join(TEST_DATA_PATH, protocol) #if channel == 'None' else os.path.join(TEST_DATA_PATH, protocol, channel)
    mat_list = sorted(glob(os.path.join(path, '*.mat'))) #if channel == 'None' else sorted(glob(os.path.join(path, '*.npy')))
    for i in SN:  
        signal_path = mat_list[i]  # Taking only the first file for simplicity
    
        print(f"Signal path: {signal_path}")  # Print signal path
        sig = sio.loadmat(signal_path) #if channel == 'None' else np.load(signal_path)
        if channel == 'None':
            sig = sig['waveform']
            len_sig = sig.shape[0]
        # print(f"Length of signal: {len_sig}")  # Print length of signal
        # print(f"seq_len: {seq_len}, sli_len: {sli_len}")
        #print(f"Signal values: {sig[:5]}")  # Print some values of the signal
            for dBs in SNR:
                noisy_sig = apply_AWGN(dBs, sig)
                len_sig = noisy_sig.shape[0]
                obs = np.stack((noisy_sig.real, noisy_sig.imag))
            # print(f"Shape of obs after stacking: {obs.shape}")  # Print shape of obs after stacking
            #print(f"Obs stack values: {obs[:, :5]}")  # Print some values of obs after stacking
                obs = np.squeeze(obs, axis=2)
            # print(f"Shape of obs after squeezing: {obs.shape}")  # Print shape of obs after squeezing
            #print(f"Obs squeeze values: {obs[:, :5]}")  # Print some values of obs after squeezing
                obs = chan2sequence(obs)
            # print(f"Shape of obs after chan2sequence: {obs.shape}")  # Print shape of obs after chan2sequence
            #print(f"Obs chan2sequence values: {obs[:10]}")  # Print some values of obs after chan2sequence
                idxs = list(range(seq_len * sli_len * 2, len_sig, seq_len * sli_len * 2))
            # print(f"Indices for splitting: {idxs}")  # Print idxs
                obs = np.split(obs, idxs)[:-1]
            # for i, segment in enumerate(obs):
            #     print(f"Length of array{i+1} after splitting :{len(segment)}")
            # print(f"Shape of obs after split:{obs}")
                for j, seq in enumerate(obs):
                    obs[j] = np.split(seq, seq_len)
                X = np.asarray(obs)
            # print(f"Shape of X after processing obs: {X.shape}")  # Print shape of X after processing obs
            #print(f"X values: {X[0, :, :5]}")  # Print some values of X
                X = torch.from_numpy(X)
                X = X.to(device)
                y = np.empty(len(idxs))
                y.fill(class_map[protocol])
                y = torch.from_numpy(y)
                y = y.to(device)
            # print(f"Y values: {y}")  # Print values of y
            
                start_time = time.time()
                pred = model(X.float())
            # save_torchviz_graph(pred, dict(model.named_parameters()), f"{protocol}_{channel}_pred")
            #save_torchviz_graph(y.mean(), dict(model.named_parameters()), f"{protocol}_{channel}_true")
            # plot_heatmap(pred.detach().cpu().numpy(), f'Prediction Heatmap for {protocol} and {channel}')
                end_time = time.time()

                print(f"Processing time for protocol {protocol}, channel {channel}: {end_time - start_time:.4f} seconds")
            # print(f"Predictions: {pred.argmax(1)}")
            # print(f"True labels: {y}")
                print(f"Accuracy: {(pred.argmax(1) == y).type(torch.float).sum().item() / len(y) * 100:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--experiment", default='3', choices=['1', '2', '3', '4'], help="Experiment mode")
    # parser.add_argument("--normalize", action='store_true', default=False, help="Use a layer norm as a first layer for CNN")
    # parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for inference")
    # parser.add_argument("--test_path", default='../data/DATASET1_1_TEST', help="Path to the dataset that will be used for testing")
    args, _ = parser.parse_known_args()
    TEST_DATA_PATH = '../data/DATASET1_1_TEST' #args.test_path
    class_map = dict(zip(PROTOCOLS, range(len(PROTOCOLS))))
    # norm_flag = '.norm' if args.normalize else ''
    device = torch.device("cuda") #if torch.cuda.is_available() and args.use_gpu else "cpu")
k_list = ['sm', 'lg']

for k in k_list:
    if k == 'lg':
        m = 128
        n = 64
    elif k == 'sm':
        m = 64
        n = 24

    model = TransformerModel(classes=len(PROTOCOLS), d_model=m * 2, seq_len=n, nlayers=2, use_pos=False)
    
    if k == 'lg':
        model.load_state_dict(torch.load(f"{TRANS_PATH}/modelNone_range_lg.pt", map_location=device)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(f"{TRANS_PATH}/modelNone_range_sm.pt", map_location=device)['model_state_dict'])        
    
    model.eval()
    model.cuda()

    for protocol in pp:
        for channel in CHANNELS:
            print(f"\nProcessing single packet for protocol {protocol}, channel {channel}")
            process_single_packet(model, class_map, seq_len=n, sli_len=m, channel=channel, protocol=protocol)
            print('\n\n\n\n')

    print(f'\nThis was for completed {k}')












# Function to save torchviz graph
# def save_torchviz_graph(output, params, filename):
#     dot = make_dot(output, params=params)
#     dot.format = 'png'
#     # Check if the 'hidden' folder exists, if not, create it
#     if not os.path.exists('hidden'):
#         os.makedirs('hidden')
#     # Save the file in the 'hidden' folder
#     dot.render(os.path.join('hidden', filename))



# def process_single_packet(model, class_map, seq_len, sli_len, channel, protocol):
#     path = os.path.join(TEST_DATA_PATH, protocol) #if channel == 'None' else os.path.join(TEST_DATA_PATH, protocol, channel)
#     mat_list = sorted(glob(os.path.join(path, '*.mat'))) #if channel == 'None' else sorted(glob(os.path.join(path, '*.npy')))
#     signal_path = mat_list[0]  # Taking only the first file for simplicity
#     print(f"Signal path: {signal_path}")  # Print signal path
#     sig = sio.loadmat(signal_path) #if channel == 'None' else np.load(signal_path)
#     if channel == 'None':
#         sig = sig['waveform']
#         len_sig = sig.shape[0]
#         print(f"Length of signal: {len_sig}")  # Print length of signal
#         print(f"seq_len: {seq_len}, sli_len: {sli_len}")
#         #print(f"Signal values: {sig[:5]}")  # Print some values of the signal
#         for dBs in SNR:
#             noisy_sig = apply_AWGN(dBs, sig)
#             len_sig = noisy_sig.shape[0]
#             obs = np.stack((noisy_sig.real, noisy_sig.imag))
#             print(f"Shape of obs after stacking: {obs.shape}")  # Print shape of obs after stacking
#             #print(f"Obs stack values: {obs[:, :5]}")  # Print some values of obs after stacking
#             obs = np.squeeze(obs, axis=2)
#             print(f"Shape of obs after squeezing: {obs.shape}")  # Print shape of obs after squeezing
#             #print(f"Obs squeeze values: {obs[:, :5]}")  # Print some values of obs after squeezing
#             obs = chan2sequence(obs)
#             print(f"Shape of obs after chan2sequence: {obs.shape}")  # Print shape of obs after chan2sequence
#             #print(f"Obs chan2sequence values: {obs[:10]}")  # Print some values of obs after chan2sequence
#             idxs = list(range(seq_len * sli_len * 2, len_sig, seq_len * sli_len * 2))
#             print(f"Indices for splitting: {idxs}")  # Print idxs
#             obs = np.split(obs, idxs)[:-1]
#             for i, segment in enumerate(obs):
#                 print(f"Length of array{i+1} after splitting :{len(segment)}")
#             print(f"Shape of obs after split:{obs}")
#             for j, seq in enumerate(obs):
#                 obs[j] = np.split(seq, seq_len)
#             X = np.asarray(obs)
#             print(f"Shape of X after processing obs: {X.shape}")  # Print shape of X after processing obs
#             #print(f"X values: {X[0, :, :5]}")  # Print some values of X
#             X = torch.from_numpy(X)
#             X = X.to(device)
#             y = np.empty(len(idxs))
#             y.fill(class_map[protocol])
#             print('l\n\n\n')
#             print(y)
#             print(y.shape)
#             print('\n\n\n')
#             y = torch.from_numpy(y)
#             y = y.to(device)
#             print(f"Y values: {y}")  # Print values of y
            
#             start_time = time.time()
#             pred = model(X.float())
#             # save_torchviz_graph(pred, dict(model.named_parameters()), f"{protocol}_{channel}_pred")
#             #save_torchviz_graph(y.mean(), dict(model.named_parameters()), f"{protocol}_{channel}_true")
#             # plot_heatmap(pred.detach().cpu().numpy(), f'Prediction Heatmap for {protocol} and {channel}')
#             end_time = time.time()

#             print(f"Processing time for protocol {protocol}, channel {channel}: {end_time - start_time:.4f} seconds")
#             print(f"Predictions: {pred.argmax(1)}")
#             print(f"True labels: {y}")
#             print(f"Accuracy: {(pred.argmax(1) == y).type(torch.float).sum().item() / len(y) * 100:.2f}%")


# TEST_DATA_PATH = '../data/DATASET1_1_TEST' #args.test_path
# class_map = dict(zip(PROTOCOLS, range(len(PROTOCOLS))))
# device = torch.device("cuda")
# model = TransformerModel(classes=len(PROTOCOLS), d_model=64 * 2, seq_len=24, nlayers=1, use_pos=False)
# model.load_state_dict(torch.load(f"{TRANS_PATH}/modelNone_range_sm.pt", map_location=device)['model_state_dict'])
# model.eval()
# model.cuda()
#     # for protocol in pp:
#     #     for channel in CHANNELS:
#     #         print(f"\nProcessing single packet for protocol {protocol}, channel {channel}")
#     #         process_single_packet(model, class_map, seq_len=24, sli_len=64, channel=channel, protocol=protocol)        
#     #         print('\n\n\n\n')
# process_single_packet(model,class_map,24,64,'None','802_11n')
# print('\n')
# print('completed')







