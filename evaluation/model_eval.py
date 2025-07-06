import numpy as np
import torch

from typing import Tuple

from librosa import frames_to_time
from librosa.util import peak_pick
from mir_eval.onset import f_measure

def evaluate(loader, model, loss_fn, dataset='test'):
    """Return average fscore, precision, recall, and loss across the given dataset."""
    print(f'evaluating the {dataset} set...')

    tolerance = loader.dataset.dataset.dsp.tolerance
    fs = loader.dataset.dataset.dsp.fs
    stride = loader.dataset.dataset.dsp.stride

    model.eval()

    fscore, precision, recall = 0.0, 0.0, 0.0
    size = 0
    with torch.no_grad():
        for (spectrogram, indices), targets in loader:
            spectrogram = spectrogram.to(model.device)
            indices = indices.to(model.device)
            targets = targets.to(model.device)
            predictions = model((spectrogram, indices))
            loss = loss_fn(predictions, targets)
            f1, pr, re = evaluate_batch(predictions, targets, tolerance, fs, stride)
            fscore += f1
            precision += pr
            recall += re
            size += 1
    fscore /= size
    precision /= size
    recall /= size

    print(f'F-score: {fscore}\nprecision: {precision}\nrecall: {recall}\n')

    model.train()

    return fscore, precision, recall, loss

def evaluate_batch(predictions, targets, tolerance, fs, stride):
    """Return the average fscore, precision, and recall of all frames the batch.

    Requires sample rate (fs) and stride to convert to back time domain.

    """
    fscore, precision, recall = 0.0, 0.0, 0.0
    for p, t in zip(predictions, targets):
        f1, pr, re = evaluate_frame(p, t, tolerance, fs, stride)
        fscore += f1
        precision += pr
        recall += re
    fscore /= predictions.shape[0]
    precision /= predictions.shape[0]
    recall /= predictions.shape[0]
    return fscore, precision, recall

def evaluate_frame(predictions, targets, tolerance, fs, stride):
    """Evaluate the current frame using mir_eval's (Bipartite matching) method.

    Requires sample rate (fs) and stride to convert to back time domain.

    """
    p = predictions.cpu().detach().numpy()
    p = peak_pick(p, 7, 7, 7, 7, 0.5, 5)
    p,= np.nonzero(p)
    p = frames_to_time(p, fs, stride)

    t = targets.cpu().detach().numpy()
    t = peak_pick(t, 7, 7, 7, 7, 0.5, 5)
    t,= np.nonzero(t)
    t = frames_to_time(t, fs, stride)

    return f_measure(t, p, tolerance)

def evaluate_frame_naive(predictions, targets):
    """Naive evaluation without a tolerance window."""
    preds = predictions.cpu().detach().numpy()
    pred_peaks = peak_pick(preds, 7, 7, 7, 7, 0.5, 5).astype(np.int32)
    preds = np.zeros(predictions.shape[0])
    np.put(preds, pred_peaks, 1.)

    targs = targets.cpu().detach().numpy()
    targ_peaks = peak_pick(targs, 7, 7, 7, 7, 0.5, 5).astype(np.int32)
    targs = np.zeros(targets.shape[0])
    np.put(targs, targ_peaks, 1.)

    sum_ = targs + preds
    diff = targs - preds

    tp = np.extract(sum_ > 1, sum_).size
    fp = np.extract(diff < 0, diff).size
    fn = np.extract(diff > 0, diff).size

    return fscore_precision_recall(tp, fp, fn)

def fscore_precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return fscore, precision, and recall."""
    return tp / max(tp + 0.5*(fp + fn), 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)
