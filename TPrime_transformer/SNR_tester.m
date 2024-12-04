function [] = SNR_tester(target_Snr,frac,~)

load('DATASET1_1_TEST/mat_ax.mat','waveform_ax');

if(frac==0)
waveform = waveform_ax;
waveform = apply_AWGN(waveform,target_Snr);
save("DATASET1_1_TEST/802_11ax/802.11ax_IQ_frame_1.mat","waveform");
end

if(frac==1)
waveform = waveform_ax(1:6800);

w = waveform()';  
total_B = 40e6;  
signal_B = 20e6; 
actual_snr = target_Snr - 10 * log10(total_B / signal_B);  % Adjust SNR based on bandwidth
waveform1 = resample(w, 2, 1);  
waveform1 = apply_AWGN(waveform1, actual_snr);

% STFT Parameters
fs = 40e6;  
nwin = 256;  
overlap = nwin * 0.5;  % Overlap between windows (55%)
win = gausswin(nwin);  

% Zero-padding and compute STFT
szeropad = [zeros(1, nwin), waveform1, zeros(1, nwin)]';  % Zero-padding the waveform
[S, F, T] = stft(szeropad, fs, 'Window', win, 'OverlapLength', overlap);  % STFT computation

matrix = imgaussfilt(abs(S), 4);

% Step 2: Edge detection (Canny method)
edges = edge(matrix, 'canny', [0.35, 0.45]);  % Apply Canny edge detection

% figure;
% imshow(edges);
% title('Edge Detection Result');

time_bins = sum(edges, 1);
freq_bins = sum(edges, 2);

time_bins_norm = time_bins / max(time_bins);  
freq_bins_norm = freq_bins / max(freq_bins); %STBT  

threshold_time = 0.3;  %STBT
threshold_freq = 0.3;  

time_bins = time_bins_norm > threshold_time; %STBT
freq_bins = freq_bins_norm > threshold_freq; 

fses = find(freq_bins == 1);
tses = find(time_bins == 1);

time_bins(min(tses):max(tses)) = 1;
freq_bins(min(fses):max(fses)) = 1;

projected_S = freq_bins .* time_bins;
projected_S = logical(projected_S);
S_hat = projected_S .* S; 

% figure
% imagesc(abs(projected_S))
% 
% figure
% imagesc(abs(S_hat));
% 
% figure
% imagesc(abs(S));

% Reconstruct the signal using inverse STFT
[signalReconFiltered, ~] = stftRecon(S_hat, fs, win, nwin, overlap);

% Downsample the reconstructed signal to its original sampling rate
waveform = resample(signalReconFiltered,1,2);
waveform = repmat(waveform,5,1);

save("DATASET1_1_TEST/802_11ax/802.11ax_IQ_frame_1.mat","waveform");
end
end