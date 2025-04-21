from train import SquareDataset, train_autoencoder, visualize_latent_space
from model import Autoencoder
import torch
from torch.utils.data import DataLoader
import os

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize model with larger latent dimension
    latent_dim = 2
    model = Autoencoder(latent_dim=latent_dim).to(device)

    # Create experiment directory with latent dimension in name
    experiment_dir = os.path.join('experiments', f'variable_location_squares_latent{latent_dim}')
    os.makedirs(experiment_dir, exist_ok=True)

    # Create dataset and dataloader
    dataset = SquareDataset('data/variable_location_squares')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the model
    history = train_autoencoder(model, train_loader, num_epochs=25, device=device, experiment_dir=experiment_dir)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(experiment_dir, 'model.pth'))
    
    # Visualize results
    visualize_latent_space(model, train_loader, device, experiment_dir)
    
    print(f'Experiment completed. Results saved in: {experiment_dir}')

if __name__ == '__main__':
    main() 