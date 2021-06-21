import torch
from torch import distributions
import torchvision as tv
from torchvision.transforms import transforms

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from acflow import RealNVP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_mnist(batch_size):
  transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.30811,))
    ]
  )

  train_set = tv.datasets.MNIST(
    '../data/', train=True, download=True, transform=transform
  )
  test_set = tv.datasets.MNIST(
    '../data/', train=False, download=True, transform=transform
  )

  train_loader = torch.utils.data.DataLoader(
    train_set, batch_size
  )
  test_loader = torch.utils.data.DataLoader(
    test_set, batch_size
  )

  return train_loader, test_loader

def train(model, train_loader, num_epochs=10):

  optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
  losses = []
  running_loss = 0

  with tqdm(total=len(train_loader)*num_epochs) as progress:
    for _ in range(num_epochs):
      for i , (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        loss = model.log_prob(x)
        loss = -loss.mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress.set_description(f'loss: {loss.item():.4f}')
        progress.update()

        if i % 100 == 99:
          losses.append(running_loss / 100)
          running_loss = 0
  
  return model, losses

def test():
  pass


if __name__ == '__main__':

  model = RealNVP(
    1, 5, 6,
    distributions.MultivariateNormal(
      torch.zeros(784).to(device), torch.eye(784).to(device)
    ),
    (1,28,28),
    device
  ).to(device)

  train_loader, test_loader = get_mnist(128)
  model, losses = train(model, train_loader, 50d)

  torch.save(
    {
    'model_state_dict': model.state_dict()
    }, 'chkpt/test.tar'
  )

  plt.plot(losses)
  plt.savefig('vis/test.png')
  # model.load_state_dict(torch.load('chkpt/test.tar')['model_state_dict'])

  samples = model.sample(100).detach().cpu().numpy()

  with open('chkpt/test_samples.pickle','wb') as out:
    pickle.dump(samples, out)