import numpy as np
import os
from PIL import Image

# Create output directory if it doesn't exist
os.makedirs('data/squares', exist_ok=True)

# Image parameters
image_size = 32
n_images = 1000

# Generate images
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
    image.save(f'data/squares/square_{i:04d}.png') 