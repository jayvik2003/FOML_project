import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load .mat file
data = scipy.io.loadmat('matlab1.mat')

# Extract src and t_out
src = data['src']
t_out = data['t_out']

# Apply sigmoid to src and t_out
src_sigmoid = sigmoid(src)
t_out_sigmoid = sigmoid(t_out)

# Convert sigmoid outputs to appropriate integers
src_int = np.round(src_sigmoid * 10).astype(int)  # Adjust the multiplier as needed
t_out_int = np.round(t_out_sigmoid * 10).astype(int)  # Adjust the multiplier as needed

# Check the shape to confirm
print("Shape of src_int:", src_int.shape)
print("Shape of t_out_int:", t_out_int.shape)

# Flatten the arrays to 1D
src_flat = src_int.flatten()
t_out_flat = t_out_int.flatten()

# Define bins and categorize the continuous values
num_bins = 10
bins = np.linspace(0, 10, num_bins + 1)  # Adjusted to match the integer range
src_binned = np.digitize(src_flat, bins) - 1  # Binning the src array
t_out_binned = np.digitize(t_out_flat, bins) - 1  # Binning the t_out array

# Compute confusion matrix
cm = confusion_matrix(src_binned, t_out_binned)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.arange(num_bins), yticklabels=np.arange(num_bins))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)    # Ensure y-axis labels are horizontal
plt.tight_layout()        # Adjust layout to make room for labels
plt.show()

# import scipy.io as sio
# import numpy as np
# import torch
# import os

# # CONFIG
# TEST_DATA_PATH = '../data/DATASET1_1_TEST'
# SNR = [0]  # Only one SNR level for simplicity
# np.random.seed(4389)  # for reproducibility


# # Reverse helper functions
# def sequence2chan(seq):
#     obs_real = seq[0::2]
#     obs_imag = seq[1::2]
#     return obs_real, obs_imag

# def reconstruct_signal(X, seq_len):
#     X = X.reshape(-1, seq_len * 2)  # Reshape to (num_sequences, seq_len * 2)
#     obs_seq = np.concatenate(X, axis=0)  # Concatenate sequences along the first axis
#     obs_real, obs_imag = sequence2chan(obs_seq)
#     noisy_sig = obs_real + 1j * obs_imag
#     return noisy_sig

# # Example usage
# if __name__ == "__main__":
#     # Load the tensor from the .mat file
#     mat_data = sio.loadmat('transformer_output.mat')
#     tensor = mat_data['t_out']

#     # Assuming tensor dimensions are correct for reshaping
#     seq_len = 24

#     # Convert numpy array to torch tensor if needed
#     tensor = torch.from_numpy(tensor)

#     # Reconstruct the signal
#     reconstructed_signal = reconstruct_signal(tensor.numpy(), seq_len)

#     # Save the signal to a .mat file
#     output_path = 'reconstructed_signal.mat'
#     sio.savemat(output_path, {'waveform': reconstructed_signal})
#     print(f"Reconstructed signal saved to {output_path}")










# # import scipy.io as sio
# # import numpy as np
# # import torch
# # import os
# # from glob import glob

# # # CONFIG
# # TEST_DATA_PATH = '../data/DATASET1_1_TEST'
# # SNR = [0]  # Only one SNR level for simplicity
# # np.random.seed(4389)  # for reproducibility

# # # Helper functions
# # def apply_AWGN(snr_dbs, sig):
# #     rms = np.sqrt(np.mean(np.abs(sig) ** 2))
# #     sig_W = rms ** 2
# #     sig_dbW = 10 * np.log10(sig_W / 1)
# #     noise_dbW = sig_dbW - float(snr_dbs)
# #     noise_var = 10 ** (noise_dbW / 10)
# #     noise_std = np.sqrt(noise_var)
# #     complex_std = noise_std * 1 / np.sqrt(2)
# #     noise_samples = np.random.normal(0, complex_std, size=sig.shape) + 1j * np.random.normal(0, complex_std, size=sig.shape)
# #     noisy_sig = sig + noise_samples
# #     return noisy_sig

# # def chan2sequence(obs):
# #     seq = np.empty((obs.size))
# #     seq[0::2] = obs[0]
# #     seq[1::2] = obs[1]
# #     return seq

# # def process_single_packet(seq_len, sli_len, protocol):
# #     path = os.path.join(TEST_DATA_PATH, protocol)
# #     mat_list = sorted(glob(os.path.join(path, '*.mat')))
# #     signal_path = mat_list[0]  # Taking only the first file for simplicity
# #     print(f"Signal path: {signal_path}")  # Print signal path
# #     sig = sio.loadmat(signal_path)['waveform']
# #     len_sig = sig.shape[0]
# #     print(f"Length of signal: {len_sig}")  # Print length of signal
# #     print(f"seq_len: {seq_len}, sli_len: {sli_len}")
# #     for dBs in SNR:
# #         noisy_sig = apply_AWGN(dBs, sig)
# #         len_sig = noisy_sig.shape[0]
# #         obs = np.stack((noisy_sig.real, noisy_sig.imag))
# #         print(f"Shape of obs after stacking: {obs.shape}")  # Print shape of obs after stacking
# #         obs = np.squeeze(obs, axis=2)
# #         print(f"Shape of obs after squeezing: {obs.shape}")  # Print shape of obs after squeezing
# #         obs = chan2sequence(obs)
# #         print(f"Shape of obs after chan2sequence: {obs.shape}")  # Print shape of obs after chan2sequence
# #         idxs = list(range(seq_len * sli_len * 2, len_sig, seq_len * sli_len * 2))
# #         print(f"Indices for splitting: {idxs}")  # Print idxs
# #         obs = np.split(obs, idxs)[:-1]
# #         for i, segment in enumerate(obs):
# #             print(f"Length of array{i+1} after splitting :{len(segment)}")
# #         print(f"Shape of obs after split:{obs}")
# #         for j, seq in enumerate(obs):
# #             obs[j] = np.split(seq, seq_len)
# #         X = np.asarray(obs)
# #         print(f"Shape of X after processing obs: {X.shape}")  # Print shape of X after processing obs
# #         X = torch.from_numpy(X)
# #         return X

# # if __name__ == "__main__":
# #     seq_len = 24
# #     sli_len = 64
# #     protocol = '802_11ax'
# #     tensor = process_single_packet(seq_len, sli_len, protocol)
# #     print(f"Final tensor shape: {tensor.shape}")
