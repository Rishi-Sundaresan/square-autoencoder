import numpy as np
import os
from PIL import Image

def generate_centered_squares(n_images=1000, image_size=32, output_dir='data/centered_squares'):
    """Generate dataset of centered squares."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(n_images):
        # Create a white background
        img = np.ones((image_size, image_size), dtype=np.uint8) * 255
        
        # Random square size between 4 and 28 pixels
        square_size = np.random.randint(4, 29)
        
        # Calculate coordinates to center the square
        start_x = (image_size - square_size) // 2
        start_y = (image_size - square_size) // 2
        
        # Draw black square
        img[start_y:start_y+square_size, start_x:start_x+square_size] = 0
        
        # Save image
        image = Image.fromarray(img)
        image.save(f'{output_dir}/square_{i:04d}.png')

def generate_variable_location_squares(n_images=2048, image_size=32, min_square_size=4, max_square_size=24, output_dir='data/variable_location_squares', margin=2):
    """Generate dataset of squares with variable sizes in different locations.
    
    Args:
        n_images: Number of images to generate
        image_size: Size of the square image
        min_square_size: Minimum size of the squares
        max_square_size: Maximum size of the squares (must be <= image_size - 2*margin)
        output_dir: Directory to save images
        margin: Minimum distance from square edges to image edges
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure max_square_size leaves enough room for positioning
    max_square_size = min(max_square_size, image_size - 2 * margin)
    
    for i in range(n_images):
        # Create a white background
        img = np.ones((image_size, image_size), dtype=np.uint8) * 255
        
        # Random square size within the specified range
        square_size = np.random.randint(min_square_size, max_square_size + 1)
        
        # Calculate maximum allowed position to keep square within bounds with margin
        max_pos = image_size - square_size - 2 * margin
        
        # Random position for the square, ensuring margin from edges
        start_x = np.random.randint(margin, max_pos + 1)
        start_y = np.random.randint(margin, max_pos + 1)
        
        # Draw black square
        img[start_y:start_y+square_size, start_x:start_x+square_size] = 0
        
        # Save image
        image = Image.fromarray(img)
        image.save(f'{output_dir}/square_{i:04d}.png')

if __name__ == '__main__':
    # Generate all datasets
    generate_centered_squares()
    generate_variable_location_squares() 