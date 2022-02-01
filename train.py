import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from dataset.dataset import OnsetDataset
from model.hyperparameters import ML, DSP
from model.model import OnsetDetector

from itertools import starmap
from functools import wraps
from time import time

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
    for epoch in range(ml.num_epochs):
        for (spectrogram, indices), targets in train_data:
            spectrogram = spectrogram.to(model.device)
            indices = indices.to(model.device)
            targets = targets.to(model.device)
            model.zero_grad()
            predictions = model((spectrogram, indices))
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch} loss: {loss}')
        validate(validation_data, model, loss_fn, optimizer, ml)

# TODO: implement validation
def validate(validation_data, model, loss_fn, optimizer, ml):
    pass

# TODO: implement testing
def test(test_data, model, loss_fn, optimizer):
    pass

# TODO: implement inference
def infer(input_data, model):
    pass

def evaluate(self, loader, model, dataset='test'):
    """Check accuracy on training & test to see how good our model is."""

    print(f'evaluating the {dataset} set')

    tolerance = loader.dataset.dataset.dsp.tolerance

    model.eval()

    with torch.no_grad():
        for (spectrogram, indices), targets in loader:
            spectrogram = spectrogram.to(model.device)
            indices = indices.to(model.device)
            targets = targets.to(model.device)
            predictions = model((spectrogram, indices))
            evaluate_batch(predictions, targets, tolerance)

    '''
        print(f"Got {num_correct} / {num_samples} with accuracy  \
              {float(num_correct)/float(num_samples)*100:.2f}")
    '''

    model.train()

def evaluate_batch(predictions, targets, tolerance):
    for i, p, t in enumerate(zip(predictions, targets)):
        print(evaluate_frame(p, t, tolerance))

def evaluate_frame(predictions, targets, tolerance):
    predictions = list(predictions)
    targets = list(targets)
    results = [0]*len(targets)
    r = {2: 0, 1: 0, -2: 0, -1: 0}
    for i in range(len(targets)):
        results[i] = sum(list(starmap(
            lambda x, y: x + y,
            zip(predictions[i-tolerance:i+tolerance],
                    targets[i-tolerance:i+tolerance]))))
        if results[i] == 0:
            predictions[i] = 0
            targets[i] = 0
        elif results[i]: pass
    return r

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
