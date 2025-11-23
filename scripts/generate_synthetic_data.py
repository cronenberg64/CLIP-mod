import os
import csv
from PIL import Image
import numpy as np

def create_synthetic_dataset(root_dir="data/synthetic", num_samples=100):
    os.makedirs(root_dir, exist_ok=True)
    images_dir = os.path.join(root_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    csv_path = os.path.join(root_dir, "train.csv")
    
    print(f"Generating {num_samples} synthetic samples in {root_dir}...")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "caption"])
        
        for i in range(num_samples):
            # Generate random image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            filename = f"image_{i:04d}.jpg"
            filepath = os.path.join(images_dir, filename)
            img.save(filepath)
            
            # Generate dummy caption
            caption = f"This is synthetic image number {i} with random noise."
            
            # Write relative path to CSV (assuming training script runs from project root)
            # But usually webdataset or csv dataset loaders expect full paths or relative to root.
            # Let's use absolute path for safety in this test, or relative to where we run.
            # OpenCLIP CSV loader typically expects 'filepath' and 'caption'.
            writer.writerow([filepath, caption])
            
    print("Dataset generation complete.")

if __name__ == "__main__":
    create_synthetic_dataset()
