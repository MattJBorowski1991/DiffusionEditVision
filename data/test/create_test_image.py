from PIL import Image, ImageDraw
import os

# Create a simple test image with shapes
img = Image.new('RGB', (256, 256), color='white')
draw = ImageDraw.Draw(img)

# Draw some shapes
draw.rectangle([50, 50, 200, 150], outline='black', width=2)
draw.ellipse([100, 100, 180, 180], outline='black', width=2)
draw.line([20, 20, 236, 236], fill='black', width=2)

# Create test directory if it doesn't exist
test_dir = os.path.join('data', 'test')
os.makedirs(test_dir, exist_ok=True)

# Save the test image
test_image_path = os.path.join(test_dir, 'test_shapes.jpg')
img.save(test_image_path)
print(f"Test image created at: {test_image_path}")
