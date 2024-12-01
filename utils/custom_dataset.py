from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform, predict_noise, include_timestep, scheduler):
        """
        Dataset for diffusion and denoising models.

        Parameters:
        -----------
        image_paths: list
            Paths to the images.
        transform: torchvision.transforms.Compose
            Transformations applied to the images.
        predict_noise: bool
            If True, predicts the noise. Otherwise, predicts the denoised image.
        include_timestep: bool
            Includes the timestep in the output.
        scheduler: DiffusionScheduler
            Instance of the DiffusionScheduler class to handle noise addition.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.predict_noise = predict_noise
        self.include_timestep = include_timestep
        self.scheduler = scheduler

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.include_timestep:
            t = np.random.randint(0, self.num_timesteps)
            noisy_image, noise = self.add_noise_fn(image, t)
            return noisy_image, noise if self.predict_noise else image, t
        else:
            noisy_image, noise = self.add_noise_fn(image)
            return noisy_image, noise if self.predict_noise else image
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.include_timestep:
            # Sample a timestep and add noise
            t = self.scheduler.sample_timesteps(batch_size=1).item()
            noisy_image, noise = self.scheduler.add_noise(image.unsqueeze(0), torch.tensor([t]))
            noisy_image, noise = noisy_image.squeeze(0), noise.squeeze(0)
            return (noisy_image, noise if self.predict_noise else image, t)
        else:
            # Add fixed or random noise for denoising tasks
            noisy_image, noise = self.scheduler.add_noise(image.unsqueeze(0), None)
            noisy_image, noise = noisy_image.squeeze(0), noise.squeeze(0)
            return (noisy_image, noise if self.predict_noise else image)