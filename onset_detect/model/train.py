import logging

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from onset_detect.dataset.dataset import OnsetDataset
from onset_detect.evaluation.model_eval import evaluate
from onset_detect.model.hyperparameters import DSP, ML
from onset_detect.model.model import OnsetDetector

MODEL_PATH = './onset_detect/model/trained_models'

log = logging.getLogger(__name__)


def train(train_data, validation_data, model, loss_fn, optimizer, ml):
    """Train the model, stopping early if it does not learn properly.

    Parameters
    ----------
    train_data : torch.utils.data.Dataloader
        The loader for the training data.
    validation_data : torch.utils.data.Dataloader
        The loader for the validation data.
    model : nn.Module
        The neural network model.
    loss_fn : nn.BCEWithLogitsLoss | nn.BCELoss
        The loss function for onset detection.
    optimizer : torch.optim.Optimizer
        The optimizer for onset detection.
    ml : ML
        The machine learning hyperparameters.

    Returns
    -------
    None.

    """
    log.info('Started training...\n')
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
        log.info(f'epoch {epoch} training loss: {loss}')

        if validate(validation_data, model, loss_fn, epoch, optimizer, ml):
            continue
        else:
            log.info(
                f'Patience Exceeded! ({model.patience} epochs with no f-score improvement). Stopping Early.'
            )
            break


def validate(validation_data, model, loss_fn, epoch, optimizer, ml) -> bool:
    """Validate the data to make sure the model is learning properly.

    If it is not learning (e.g. vanishing gradient or convergence)
    return False to indicate that training should stop.

    Parameters
    ----------
    validation_data : torch.utils.data.Dataloader
        The Dataloader for the validation data.
    model : nn.Module
        The model to validate.
    loss_fn : nn.BCEWithLogitsLoss | nn.BCELoss
        The loss function for onset detection.
    optimizer : torch.optim.Optimizer
        The optimizer for onset detection.
    ml : ML
        The machine learning hyperparameters.

    Returns
    -------
    bool
        Whether or not training should continue.

    """
    fscore, precision, recall, loss = evaluate(
        validation_data, model, loss_fn, dataset='dev'
    )
    log.info(f'epoch {epoch} validation loss: {loss}')

    if fscore > model.hi_score:
        model.hi_score = fscore
        model.no_improve = 0
        torch.save(model.state_dict(), f'{MODEL_PATH}/f_{fscore}.pt')
    else:
        model.no_improve += 1

    if model.no_improve > model.patience:
        return False
    return True


def main():
    """Set the hyperparameters, load and split the data, and train the model."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ml = ML()
    dsp = DSP()
    dataset = OnsetDataset(ml=ml, dsp=dsp)
    # (spec, ind), targ = dataset.__getitem__(0)
    # log.info(spec.shape, ind.shape, targ.shape)
    train_data, test_data, validation_data = dataset.split_train_test_dev(2)
    train_loader = DataLoader(
        train_data,
        batch_size=ml.batch_size,
        num_workers=ml.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=ml.batch_size, num_workers=ml.num_workers, pin_memory=True
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=ml.batch_size,
        num_workers=ml.num_workers,
        pin_memory=True,
    )
    model = OnsetDetector(**ml.__dict__, device=device).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=ml.learning_rate)

    train(train_loader, validation_loader, model, loss_fn, optimizer, ml)
    evaluate(test_loader, model)


if __name__ == '__main__':
    main()
