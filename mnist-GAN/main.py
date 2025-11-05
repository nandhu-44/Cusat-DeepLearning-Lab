import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

# Create output directory
os.makedirs('./outputs/generated', exist_ok=True)

# Hyperparameters
latent_dim = 100
num_classes = 10
img_size = 28
channels = 1
batch_size = 64
lr = 0.0002
epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat([noise, label_embedding], dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(channels * img_size * img_size + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        img_flat = img.view(img.size(0), -1)
        d_input = torch.cat([img_flat, label_embedding], dim=1)
        validity = self.model(d_input)
        return validity

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Training loop
print(f"Training on {device}")
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size_current = imgs.size(0)
        
        # Adversarial ground truths
        valid = torch.ones(batch_size_current, 1).to(device)
        fake = torch.zeros(batch_size_current, 1).to(device)
        
        # Real images and labels
        real_imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Train Generator
        optimizer_G.zero_grad()
        
        # Sample noise and labels
        z = torch.randn(batch_size_current, latent_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size_current,)).to(device)
        
        # Generate images
        gen_imgs = generator(z, gen_labels)
        
        # Generator loss
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Real images loss
        real_validity = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(real_validity, valid)
        
        # Fake images loss
        fake_validity = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(fake_validity, fake)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
    
    # Save sample images every 10 epochs
    if epoch % 10 == 0:
        generator.eval()
        with torch.no_grad():
            sample_labels = torch.arange(0, 10).to(device)
            z = torch.randn(10, latent_dim).to(device)
            gen_imgs = generator(z, sample_labels)
            
            fig, axes = plt.subplots(1, 10, figsize=(15, 2))
            for j in range(10):
                axes[j].imshow(gen_imgs[j, 0].cpu().numpy(), cmap='gray')
                axes[j].axis('off')
                axes[j].set_title(f'{j}')
            plt.savefig(f'./outputs/generated/epoch_{epoch}.png')
            plt.close()
        generator.train()

# Save trained model
torch.save(generator.state_dict(), './outputs/generator.pth')
print("\nTraining complete! Model saved to ./outputs/generator.pth")

# Function to generate specific digits
def generate_digit(digit, num_samples=1):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.tensor([digit] * num_samples).to(device)
        gen_imgs = generator(z, labels)
        
        for i in range(num_samples):
            img = gen_imgs[i, 0].cpu().numpy()
            plt.figure(figsize=(3, 3))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f'Generated digit: {digit}')
            plt.savefig(f'./outputs/generated/digit_{digit}_sample_{i}.png', bbox_inches='tight')
            plt.close()
            print(f"Generated digit {digit} saved to ./outputs/generated/digit_{digit}_sample_{i}.png")


while True:
    user_input = input("Enter a digit (0-9) to generate or 'exit' to quit: ")
    if user_input.lower() == 'exit':
        break
    elif user_input.isdigit() and 0 <= int(user_input) <= 9:
        generate_digit(int(user_input), num_samples=1)
    else:
        print("Invalid input. Please enter a digit between 0 and 9 or 'exit'.")
