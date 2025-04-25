import torch
import torchdrift.data.functional as drift_f
from prepare_data import DataPreparation
import os
import torchvision.utils as vutils

if __name__ == "__main__":
    data_dir = "inaturalist_12K"
    batch_size = 32
    output_dir = "drifted_blur_images"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare validation loader (as a source of "clean" data)
    data_prep = DataPreparation(data_dir, batch_size=batch_size, val_split=0.2)
    _, val_loader, _ = data_prep.get_data_loaders()

    # Simulate drift by applying Gaussian blur to a batch of validation images
    for imgs, labels in val_loader:
        # imgs: [batch_size, 3, 224, 224]
        drifted_imgs = drift_f.gaussian_blur(imgs, severity=1)
        print("Simulated drifted (blurred) batch shape:", drifted_imgs.shape)
        # Save each image in the batch
        for i, img in enumerate(drifted_imgs):
            # Denormalize if needed, here we assume [0,1] range
            vutils.save_image(img, os.path.join(output_dir, f"drifted_blur_{i}.png"))
        break  # Only simulate one batch for demonstration

    print(f"Drifted (blurred) images saved to folder: {output_dir}")