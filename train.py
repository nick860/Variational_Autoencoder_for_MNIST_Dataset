import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from autoencoder import *
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

def train_model(model, optimizer, criterion, train_loader, epochs, input_dim=784):
    """
    Train the VariationalAutoEncoder model
    param model: VariationalAutoEncoder model
    param optimizer: torch.optim
    param criterion: torch.nn
    param train_loader: torch.utils.data.DataLoader
    param epochs: int
    param input_dim: int
    return: VariationalAutoEncoder
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.view(-1, input_dim).to(device) # flatten image
            x_reconstructed, mu, sigma = model(x)
            loss = criterion(x_reconstructed, x) - 0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) # KL divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader.dataset)}")
    return model

def configiratoin():
    """
    Configuration for the model
    return: VariationalAutoEncoder
    """
    input_dim = 784 # 28x28 the size of MNIST images
    hidden_dim = 400
    latent_dim = 20
    batch_size = 128
    epochs = 10
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    # load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    mnist_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    # model
    model = VariationalAutoEncoder(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='sum')

    # training
    model = train_model(model, optimizer, criterion, mnist_train, epochs, input_dim)
    torch.save(model.state_dict(), 'model.pth')
    return model

def get_mu_sigma(model):
    """
    the function will return the mu and sigma for each digit
    param model: VariationalAutoEncoder
    return: dict, dict - encoding_digits, images
    """
    images = {i : [] for i in range(10)}
    idx = 0
    dataset0 = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    dataset = DataLoader(dataset0, batch_size=1, shuffle=True)
    for x, y in dataset:
        images[y.item()].append(x)
    
    encoding_digits = {i : [] for i in range(10)}
    for d in range(10):
        for i in range(100):
            with torch.no_grad():
                mu, sigma = model.encode(images[d][i].view(1, 784)) # flatten image, the 1 is the batch size because the model expects a batch
            encoding_digits[d].append((mu, sigma))

    return encoding_digits, images

def genrate_new_image(model, encoding_digits, digit_to_generate, images):
    """
    generate new image for the given digit using the model and encoding_digits
    param model: VariationalAutoEncoder
    param encoding_digits: dict
    param digit_to_generate: int
    param images: dict
    return: None
    """
    with torch.no_grad():
        #num_img = random.randint(0, 99)  # you can use it to generate random image similar to some other random image
        #img = images[digit_to_generate][num_img]
        # show image
        #img = img.view(28, 28).numpy()
        #plt.imshow(img, cmap='gray')
        #plt.show()
        # compute the mean mu and mean sigma4
        mu = torch.stack([enc[0] for enc in encoding_digits[digit_to_generate]]).mean(0) # find general mu and sigma for the digit
        sigma = torch.stack([enc[1] for enc in encoding_digits[digit_to_generate]]).mean(0)
        #mu, sigma = encoding_digits[digit_to_generate][num_img]
        epsilon = torch.randn_like(sigma)  # the magic of the VAE, sample from normal distribution
        z = mu + sigma * epsilon
        new_image = model.decode(z).view(28, 28)
        # show the image without saving
        new_image = new_image.cpu().numpy()
        plt.imshow(new_image, cmap='gray')
        plt.show()

if __name__ == "__main__":
    model = VariationalAutoEncoder(784, 400, 20)
    # train the model
    #model = configiratoin() #- if you don't have the model.pth file you may train the model before
    # load the model
    model.load_state_dict(torch.load('model.pth'))
    # get mu and sigma
    encoding_digits, images = get_mu_sigma(model)
    # generate new image
    while True:
        digit_to_generate = int(input("Enter digit to generate: "))
        genrate_new_image(model, encoding_digits, digit_to_generate, images)
