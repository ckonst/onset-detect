import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from datasets import OnsetDataset
from hyperparameters import ML, DSP

# TODO: Maxpooling ?
class OnsetDetector(nn.Module):
    """Onset detection model for automatic rhythm game mapping."""

    def __init__(self, **kwargs):
        """Useful docstring goes here."""
        super(OnsetDetector, self).__init__()
        self.__dict__.update(**kwargs)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=(3, 3))
        self.bidirectional = True
        self.D = 2 if self.bidirectional else 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
        self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear((self.input_size * self.D * self.hidden_size) + 1, self.input_size)
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0, x1 = x
        x0 = self.relu(self.bn(self.conv(x0)))
        x0 = self.relu(self.bn(self.conv(x0)))
        x0 = x0.squeeze(1).flatten(start_dim=-2, end_dim=-1)
        x0 = x0.view(x0.size(0), x0.size(1) // self.input_size, self.input_size)
        h0 = torch.zeros(self.num_layers * self.D, x0.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.D, x0.size(0), self.hidden_size).to(self.device)
        x0, _ = self.lstm(x0, (h0, c0))
        x0 = self.dropout(x0)
        x = torch.cat((x0.flatten(start_dim=1), x1), dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        return x

# FIXME: very slow 1st epoch
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
def validate(validation_data, model, loss_fn, optimizer, ml):
    pass

# TODO: implement inference
def infer(input_data, model):
    pass

def check_accuracy(loader, model, tolerance, dataset='train'):
    """Check accuracy on training & test to see how good our model is."""
    print(f'checking accuracy on the {dataset} set')

    num_correct = 0
    num_samples = 0
    tolerance = 480
    model.eval()

    with torch.no_grad():
        for (spectrogram, indices), y in loader:
            spectrogram = spectrogram.to(model.device)
            indices = indices.to(model.device)
            y = y.to(model.device)

            scores = model((spectrogram, indices))
            _, predictions = scores.max(1)
            num_correct += (predictions <= y + tolerance or predictions >= y - tolerance).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy  \
              {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ml = ML()
    train_data = OnsetDataset(**ml.__dict__)
    train_loader = DataLoader(train_data, batch_size=ml.batch_size,
                                     num_workers=ml.num_workers)
    model = OnsetDetector(**ml.__dict__, device=device).to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=ml.learning_rate)

    train(train_loader, model, loss_fn, optimizer, ml)
    test(train_loader, model, loss_fn, optimizer, ml)

if __name__ == '__main__':
    main()