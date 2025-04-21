import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
from model import Autoencoder
from train import SquareDataset, train_autoencoder

def evaluate_latent_dimensions():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Parameters
    latent_dims = range(1, 6)  # Test latent dimensions 1 through 5
    num_epochs = 50
    results = []

    # Create dataset and dataloader
    dataset = SquareDataset('data/variable_location_squares')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train models with different latent dimensions
    for latent_dim in latent_dims:
        print(f'\nTraining model with latent dimension {latent_dim}')
        
        # Create experiment directory
        experiment_dir = os.path.join('experiments', f'latent_dim_{latent_dim}')
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Initialize and train model
        model = Autoencoder(latent_dim=latent_dim).to(device)
        history = train_autoencoder(model, train_loader, num_epochs=num_epochs, device=device, experiment_dir=experiment_dir)
        
        # Save final loss
        final_loss = history['loss'][-1]
        results.append((latent_dim, final_loss))
        
        # Save model
        torch.save(model.state_dict(), os.path.join(experiment_dir, 'model.pth'))
        
        print(f'Latent dimension {latent_dim}: Final loss = {final_loss:.6f}')

    # Plot results
    latent_dims, losses = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims, losses, 'bo-')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Average Loss')
    plt.title('Model Performance vs Latent Dimension')
    plt.grid(True)
    
    # Save plot
    plt.savefig('latent_dim_comparison.png')
    plt.close()

if __name__ == '__main__':
    evaluate_latent_dimensions() 