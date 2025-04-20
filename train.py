import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from model import Autoencoder
import numpy as np
from scipy import stats

class SquareDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Extract square size from the image
        img_array = np.array(image)
        black_pixels = np.where(img_array == 0)
        if len(black_pixels[0]) > 0:
            square_size = max(
                np.max(black_pixels[0]) - np.min(black_pixels[0]),
                np.max(black_pixels[1]) - np.min(black_pixels[1])
            ) + 1
        else:
            square_size = 0
            
        image = self.transform(image)
        return image, torch.tensor(square_size, dtype=torch.float32)

def train_autoencoder(model, train_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, sizes in train_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            reconstruction, latent = model(batch)
            loss = criterion(reconstruction, batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # Save a sample reconstruction
        if (epoch + 1) % 5 == 0:
            save_sample_reconstruction(model, train_loader, epoch + 1, device)

def save_sample_reconstruction(model, train_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        # Get a sample batch
        images, _ = next(iter(train_loader))
        sample_img = images[0].to(device)
        
        # Get reconstruction
        reconstruction, _ = model(sample_img.unsqueeze(0))
        
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        # Plot original and reconstruction
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(sample_img.cpu().squeeze(), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(reconstruction.cpu().squeeze(), cmap='gray')
        plt.title('Reconstruction')
        plt.axis('off')
        
        plt.savefig(f'images/reconstruction_epoch_{epoch}.png')
        plt.close()

def visualize_latent_space(model, train_loader, device, num_samples=5):
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        sample_batch = next(iter(train_loader))
        images = sample_batch[0][:num_samples].to(device)
        
        # Get latent representations
        _, latents = model(images)
        
        # Create a figure
        fig = plt.figure(figsize=(15, 3 * num_samples))
        
        for idx in range(num_samples):
            # Plot original image
            plt.subplot(num_samples, 2, 2*idx + 1)
            plt.imshow(images[idx].cpu().squeeze(), cmap='gray')
            plt.title(f'Original Image {idx+1}')
            plt.axis('off')
            
            # Display latent representation as text
            plt.subplot(num_samples, 2, 2*idx + 2)
            latent = latents[idx].cpu().numpy()
            plt.text(0.1, 0.5, f'Latent Vector:\n{latent}', 
                    fontsize=10, family='monospace',
                    verticalalignment='center')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('latent_visualization.png')
        plt.close()

def plot_size_vs_latent(model, train_loader, device):
    model.eval()
    all_sizes = []
    all_latents = []
    
    with torch.no_grad():
        for batch, sizes in train_loader:
            batch = batch.to(device)
            _, latents = model(batch)
            # Take first dimension if latent space is > 1D
            first_latent_dim = latents[:, 0].cpu().numpy()
            all_latents.extend(first_latent_dim)
            all_sizes.extend(sizes.numpy())
    
    # Convert to numpy arrays
    all_sizes = np.array(all_sizes)
    all_latents = np.array(all_latents)
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Create scatter plot with line of best fit
    plt.figure(figsize=(10, 6))
    plt.scatter(all_sizes, all_latents, alpha=0.5, label='Data points')
    
    # Calculate line of best fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_sizes, all_latents)
    line_x = np.array([min(all_sizes), max(all_sizes)])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red', label=f'Line of best fit (R² = {r_value**2:.3f})')
    
    plt.xlabel('Square Size (pixels)')
    plt.ylabel('Latent Dimension Value')
    plt.title('Square Size vs Latent Representation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/size_vs_latent.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create dataset and dataloader
    dataset = SquareDataset('data/squares')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = Autoencoder(latent_dim=1).to(device)

    # Train the model
    train_autoencoder(model, train_loader, num_epochs=100, device=device)

    # Save the trained model
    torch.save(model.state_dict(), 'square_autoencoder.pth')
    
    # Visualize some examples with their latent representations
    visualize_latent_space(model, train_loader, device)
    
    # Plot square size vs latent value relationship
    plot_size_vs_latent(model, train_loader, device)

if __name__ == '__main__':
    main() 