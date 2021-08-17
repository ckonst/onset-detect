import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from datasets import OnsetDataset
from hyperparameters import ML, DSP
from model import OnsetDetector

def train(train_data, model, loss_fn, optimizer, ml):
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
        validate(train_data, model, loss_fn, optimizer, ml)

# TODO: implement testing
def test(test_data, model, loss_fn, optimizer, ml):
    pass

# TODO: implement validation
def validate(validation_data, model, loss_fn, optimizer):
    pass

# TODO: implement inference
def infer(input_data, model):
    pass

def evaluate(loader, model, dataset='test'):
    """Check accuracy on training & test to see how good our model is."""

    print(f'evaluating the {dataset} set')

    results = {'T+': 0, 'F+': 0, 'T-': 0, 'F-': 0}
    tolerance = loader.dataset.dsp.tolerance

    model.eval()

    with torch.no_grad():
        for (spectrogram, indices), targets in loader:
            spectrogram = spectrogram.to(model.device)
            indices = indices.to(model.device)
            targets = targets.to(model.device)

            predictions = model((spectrogram, indices))

    '''
        print(f"Got {num_correct} / {num_samples} with accuracy  \
              {float(num_correct)/float(num_samples)*100:.2f}")
    '''

    model.train()

def main():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    ml = ML()
    dsp = DSP()
    train_data = OnsetDataset(ml=ml, dsp=dsp)
    train_loader = DataLoader(train_data, batch_size=ml.batch_size,
                                     num_workers=ml.num_workers)
    model = OnsetDetector(**ml.__dict__, device=device).to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=ml.learning_rate)

    train(train_loader, model, loss_fn, optimizer, ml)
    test(train_loader, model, loss_fn, optimizer, ml)

if __name__ == '__main__':
    main()