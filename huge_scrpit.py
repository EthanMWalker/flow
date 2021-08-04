import torch
from torch import distributions
import torchvision as tv
from torchvision.transforms import transforms

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from acflow import RealNVP

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



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

def train(model, train_loader, lr, num_epochs=10, save_iters=5):

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  losses = []
  running_loss = 0

  with tqdm(total=len(train_loader)*num_epochs) as progress:
    for epoch in range(num_epochs):
      for i , (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        loss = model.log_prob(x)
        loss = -loss.mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress.set_description(f'l: {model.num_layers} lr: {lr:.8} e: {epoch} l: {loss.item():.2f}')
        progress.update()

        if i % 100 == 99:
          losses.append(running_loss / 100)
          running_loss = 0

      if epoch % save_iters == 0:
        torch.save(
          {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
          }, f'chkpt/mnist_{model.num_layers}_{lr:.8}_{epoch}.tar'
        )
  
  return model, losses


if __name__ == '__main__':
  train_loader, test_loader = get_mnist(128)

  for layers in [2**i for i in [0,1,2,3,4,5,6]]:
    for lr in [5e-8, 1e-6, 1e-5]:
      if layers < 8:
        n_epochs = 600
      else:
        n_epochs = 150
      
      model = RealNVP(
        1, 10, layers,
        distributions.MultivariateNormal(
          torch.zeros(784).to(device), torch.eye(784).to(device)
        ),
        (1,28,28),
        device
      ).to(device)

      model, losses = train(model, train_loader, lr, n_epochs, 10)

      torch.save(
        {
        'model_state_dict': model.state_dict()
        }, f'chkpt/mnist_{layers}_{lr:.8}.tar'
      )

      # model.load_state_dict(torch.load('chkpt/test.tar')['model_state_dict'])

      samples = model.sample(100).detach().cpu().numpy()
      out_dict = {
        'sample': samples,
        'losses': losses
      }

      with open(f'chkpt/mnist_samples_{layers}_{lr:.8}.pickle','wb') as out:
        pickle.dump(out_dict, out)