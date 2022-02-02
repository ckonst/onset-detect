import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from dataset.dataset import OnsetDataset
from model.hyperparameters import ML, DSP
from model.model import OnsetDetector

from functools import wraps
from time import time

from librosa import frames_to_time
from librosa.util import peak_pick
from mir_eval.onset import f_measure
import numpy as np

def measure(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {} ms'.format((end-start) * 1000))
        return result
    return wrapper

def train(train_data, validation_data, model, loss_fn, optimizer, ml):
    print('Started training...')
    for epoch in range(ml.num_epochs):
        for i, ((spectrogram, indices), targets) in enumerate(train_data):
            spectrogram = spectrogram.to(model.device)
            indices = indices.to(model.device)
            targets = targets.to(model.device)
            model.zero_grad()
            predictions = model((spectrogram, indices))
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch} loss: {loss}')

        if validate(validation_data, model, loss_fn, optimizer, ml):
            continue
        else:
            print(f'Patience Exceeded! ({model.ml.patience} epochs with no f-score improvement). Stopping Early.')
            break

#TODO: save the best model to the disk.
def validate(validation_data, model, loss_fn, optimizer, ml):
    """
    Validate the data to make sure the model is learning properly.
    If it is not learning (e.g. vanishing gradient or convergence) return False to indicate that training should stop.

    Parameters
    ----------
    validation_data : torch.utils.data.Dataloader
        The Dataloader for the validation data.
    model : nn.Module
        The model to validate.
    loss_fn : nn.BCEWithLogitsLoss
        The loss function for onset detection.
    optimizer : optim.Adam
        The optimizer for onset detection.
    ml : Hyperparameters
        The machine learning hyperparameters.

    Returns
    -------
    bool
        Whether or not training should continue.

    """
    fscore, precision, recall = evaluate(validation_data, model, dataset='dev')

    if fscore > model.hi_score:
        model.hi_score = fscore
        model.no_improve = 0
    else:
        model.no_improve += 1

    if model.no_improve > model.patience:
        return False
    return True

def evaluate(loader, model, dataset='test'):
    """Check accuracy on training & test to see how good our model is."""

    print(f'evaluating the {dataset} set')

    tolerance = loader.dataset.dataset.dsp.tolerance
    fs = loader.dataset.dataset.dsp.fs
    stride = loader.dataset.dataset.dsp.stride

    model.eval()

    fscore, precision, recall = (0., 0., 0.)
    with torch.no_grad():
        for (spectrogram, indices), targets in loader:
            spectrogram = spectrogram.to(model.device)
            indices = indices.to(model.device)
            targets = targets.to(model.device)
            predictions = model((spectrogram, indices))
            f, p, r = evaluate_batch(predictions, targets, tolerance, fs, stride)
            fscore += f
            precision += p
            recall += r

    fscore /= len(dataset)
    precision /= len(dataset)
    recall /= len(dataset)

    print(f"F-score: {fscore}\n\
          precision: {precision}\n\
             recall: {recall}")

    model.train()

    return fscore, precision, recall

def evaluate_batch(predictions, targets, tolerance, fs, stride):
    fscore, precision, recall = (0., 0., 0.)
    for p, t in zip(predictions, targets):
        f1, pr, re = evaluate_frame(p, t, tolerance, fs, stride)
        fscore += f1
        precision += pr
        recall += re
    return fscore, precision, recall


def evaluate_frame(predictions, targets, tolerance, fs, stride):
    p = predictions.cpu().detach().numpy()
    p = peak_pick(p, 7, 7, 7, 7, 0.5, 5)
    p,= np.nonzero(p)
    p = frames_to_time(p, fs, stride)

    t = targets.cpu().detach().numpy()
    t = peak_pick(t, 7, 7, 7, 7, 0.5, 5)
    t,= np.nonzero(t)
    t = frames_to_time(t, fs, stride)

    return f_measure(t, p, tolerance)

# TODO: implement testing
def test(test_data, model, loss_fn, optimizer, ml):
    pass

# TODO: implement inference
def infer(input_data, model):
    pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    ml = ML()
    dsp = DSP()
    dataset = OnsetDataset(ml=ml, dsp=dsp)
    train_data, test_data, validation_data = dataset.split_train_test_dev(2)
    train_loader = DataLoader(train_data, batch_size=ml.batch_size, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=ml.batch_size, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=ml.batch_size, pin_memory=True)
    model = OnsetDetector(**ml.__dict__, device=device).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=ml.learning_rate)

    train(train_loader, validation_loader, model, loss_fn, optimizer, ml)
    test(test_loader, model, loss_fn, optimizer, ml)

#%%

if __name__ == '__main__':
    main()
