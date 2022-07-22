# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

bs = 100
# # MNIST Dataset
# train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

class CustomScalarDataset(Dataset):
    def __init__(self, data_size=10000):
        # self.scalars = torch.randint(low=1, high=7, size=(data_size, 1),dtype=torch.float32)
        self.scalars = torch.rand(size=(data_size, 1),dtype=torch.float32)

    def __len__(self):
        return len(self.scalars)

    def __getitem__(self, idx):
        return self.scalars[idx]

train_dataset = CustomScalarDataset(data_size=100000)
test_dataset = CustomScalarDataset(data_size=1000)

# # Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return self.fc6(h)
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 1))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# build model
vae = VAE(x_dim=1, h_dim1= 512, h_dim2=256, z_dim=2)
if torch.cuda.is_available():
    vae.cuda()

vae

optimizer = optim.Adam(vae.parameters(),lr=0.001)
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    MSE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + 0.05*KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, 5):
    train(epoch)
    test()

with torch.no_grad():
    # for purely random sampling
    # z = torch.randn(64, 2).cuda()
    # sample = vae.decoder(z).cuda()
    # save_image(sample.view(64, 1, 28, 28), '/home/azureuser/Pictures/sample_' + '.png')

    # for grid sampling
    n_rows = 16
    values = torch.linspace(-4, 4, steps=n_rows)
    grid_x, grid_y = torch.meshgrid(values, values, indexing='ij')
    grid_x = grid_x.unsqueeze(dim=2)
    grid_y = grid_y.unsqueeze(dim=2)
    z = torch.cat((grid_x, grid_y), dim=2)
    z = z.reshape(-1, 2).cuda()
    sample = vae.decoder(z).cuda()
    grid = sample.view(n_rows, n_rows)
    # grid = make_grid(sample, nrow=n_rows)
    plt.imshow(grid.cpu(), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig('/home/azureuser/Pictures/heat_' + '.png')
    # save_image(grid, '/home/azureuser/Pictures/sample_uniform' + '.png')