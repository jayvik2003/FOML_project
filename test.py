import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as conf_mat
from preprocessing.TPrime_dataset import TPrimeDataset_Transformer
from model_transformer import TransformerModel, TransformerModel_v2

def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def validate_epoch(dataloader, model, loss_fn, Nclasses):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    conf_matrix = np.zeros((Nclasses, Nclasses))
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_cpu = y.to('cpu')
            pred_cpu = pred.to('cpu')
            conf_matrix += conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(Nclasses)))
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )
    return test_loss, correct, conf_matrix

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--snr_db", nargs='+', default=[30], help="SNR levels to be considered during testing.")
    parser.add_argument("--wchannel", default=None, help="Wireless channel applied, it can be TGn, TGax, Rayleigh, relative or random.")
    parser.add_argument('--raw_path', default='../data/DATASET1_1_TEST', help='Path where raw signals are stored.')
    parser.add_argument('--postfix', default='', help='Postfix to append to dataset file.')
    parser.add_argument("--cp_path", default='./model_cp', help='Path to the checkpoint to load the model from.')
    parser.add_argument("--cls_token", action="store_true", default=False, help="Use the Transformer v2")
    parser.add_argument("--dataset_ratio", default=1.0, type=float, help="Portion of the dataset used for testing.")
    parser.add_argument("--Slice_length", type=int, default=128, help="Slice length in which a sequence is divided.")
    parser.add_argument("--Sequence_length", type=int, default=64, help="Sequence length to input to the transformer.")
    parser.add_argument("--Positional_encoder")
    args, _ = parser.parse_known_args()
    args.wchannel = args.wchannel if args.wchannel != 'None' else None
    args.Positional_encoder = args.Positional_encoder in {'True', 'true'}
    postfix = '' if not args.cls_token else '_v2'
    args.cp_path = args.cp_path + postfix

    protocols = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
    ds_test = TPrimeDataset_Transformer(protocols=protocols, ds_type='test', file_postfix=args.postfix, ds_path=args.raw_path, snr_dbs=args.snr_db, seq_len=args.Sequence_length, slice_len=args.Slice_length, slice_overlap_ratio=0, raw_data_ratio=args.dataset_ratio, override_gen_map=False, apply_wchannel=args.wchannel, transform=chan2sequence)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_info = ds_test.info()
    Nclass = ds_info['nclasses']

    test_dataloader = DataLoader(ds_test, batch_size=128, shuffle=False)
    model_class = TransformerModel if not args.cls_token else TransformerModel_v2

    model = model_class(classes=Nclass, d_model=2*args.Slice_length, seq_len=args.Sequence_length, nlayers=2, use_pos=args.Positional_encoder).to(device)
    
    checkpoint = torch.load(os.path.join(args.cp_path, 'model{}_{}_lg.pt' if args.Sequence_length == 64 else 'model{}_{}_sm.pt'.format(args.wchannel, args.snr_db[0])))
    model.load_state_dict(checkpoint['model_state_dict'])

    loss_fn = nn.NLLLoss()
    test_loss, test_acc, conf_matrix = validate_epoch(test_dataloader, model, loss_fn, Nclasses=Nclass)
    
    fig = plt.figure(figsize=(8,8))
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[np.newaxis]
    plt.imshow(conf_matrix, interpolation='none', cmap=plt.cm.Blues)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(0, 1)
    tick_marks = np.arange(Nclass)
    plt.xticks(tick_marks, protocols)
    plt.yticks(tick_marks, protocols)
    plt.tight_layout()
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"Confusion matrix: {args.snr_db[0]} dBs, channel: {args.wchannel}, slice: {args.Slice_length}, seq.: {args.Sequence_length}")
    plt.show()
