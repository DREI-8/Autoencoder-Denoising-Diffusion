import kagglehub
import os
import random
from glob import glob
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from utils.diffusion_scheduler import DiffusionScheduler
from utils.custom_dataset import CustomDataset
import matplotlib.pyplot as plt

class HandleDataset():
    def __init__(self, kaggle_path, folder_name = ''):
        """
        Load a dataset from Kaggle. All images (in jpg) in the folder_name directory will be loaded (recursively).
        You can use folder_name = 'folder/subfolder' to load only a specific subfolder.

        Parameters:
        -----------
        kaggle_path: str
            Path to Kaggle dataset
        folder_name: str
            Name of the folder to load images from. Default is '' (load all images).
        """

        try:
            self.path = kagglehub.dataset_download(kaggle_path)
        except Exception as e:
            raise RuntimeError("Failed to load dataset") from e
        
        self.image_paths = glob(os.path.join(self.path, folder_name, '**', '*.jpg'), recursive=True)
        self.denoising_configured = False
    
    def plot_sample(self):
        """
        Plot a sample from the dataset (5 random images).
        """
        sample_images = random.sample(self.image_paths, 5)

        _, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, ax in enumerate(axes.flat):
            img = Image.open(sample_images[i])
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def infos(self):
        """
        Print information about the dataset.
        """
        print(f"Dataset path: {self.path}")
        print(f"Number of images: {len(self.image_paths)}")
        sample_image = Image.open(self.image_paths[0])
        print(f"Sample image shape: {sample_image.size} with {len(sample_image.getbands())} channels")

    def split(self, train_size = 0.8, valid_size = 0.1):
        """
        Split the dataset into train, validation and test sets.

        Parameters:
        -----------
        train_size: float
            Size of the training set. Default is 0.8.
        valid_size: float
            Size of the validation set. Default is 0.1.

        Returns:
        --------
        train_set: list
            List of training images paths.
        valid_set: list
            List of validation images paths.
        test_set: list
            List of test images paths.
        """
        random.shuffle(self.image_paths)
        train_size = int(len(self.image_paths) * train_size)
        valid_size = int(len(self.image_paths) * valid_size)

        self.train_set = self.image_paths[:train_size]
        self.valid_set = self.image_paths[train_size:train_size+valid_size]
        self.test_set = self.image_paths[train_size+valid_size:]
    
    def configure_denoising(self, predict_noise=True, include_timestep=True, num_timesteps=1000, noise_intensity=None):
        """
        Configure the dataset for diffusion or denoising tasks. Use include_timestep=False for denoising tasks
        and specify the noise intensity with a float or a tuple (range). Use include_timestep=True for diffusion tasks,
        and you can specify the number of timesteps. Noise intensity will not be used for diffusion tasks.
        If noise_intensity is None and include_timestep=False, a default value of 0.1 will be used.

        Parameters:
        -----------
        predict_noise: bool
            If True, the model predicts the noise. Otherwise, it predicts the denoised image.
        include_timestep: bool
            If True, includes the timestep in the output.
        num_timesteps: int
            Number of diffusion steps.
        noise_intensity: float or tuple(float, float)
            - If float, a fixed noise intensity will be used.
            - If tuple, a random noise intensity will be drawn from this range.
            - Default is None for diffusion tasks.
        """
        self.predict_noise = predict_noise
        self.include_timestep = include_timestep
        self.denoising_configured = True

        self.scheduler = DiffusionScheduler(num_timesteps, noise_intensity)        
    
    def prepare(self, augment=True, target_size=(64, 64), batch_size=32, shuffle=True, custom_transforms_train=None, custom_transforms_val_test=None):
        """
        Prepare the dataset for training. Images are resized to target_size, normalized (with ImageNet mean and std) and loaded into DataLoader.
        If augment is True, random horizontal flip and rotation are applied to the training set. 
        You can also pass custom transformations that will override the default ones (augment and target_size are ignored).

        Parameters:
        -----------
        augment: bool
            Apply data augmentation to the training set. Default is True.
        target_size: tuple
            Target size for the images. Default is (64, 64).
        batch_size: int
            Batch size for DataLoader. Default is 32.
        shuffle: bool
            Shuffle the DataLoader, only for the train set. Default is True.
        custom_transforms_train: list
            Custom transformations for the training set. Default is None.
        custom_transforms_val_test: list
            Custom transformations for the validation and test sets. Default is None.

        Returns:
        --------
        train_loader: DataLoader
            DataLoader for the training set.
        valid_loader: DataLoader
            DataLoader for the validation set.
        test_loader: DataLoader
            DataLoader for the test set.
        """

        if not self.denoising_configured:
            print("Warning: Denoising not configured. You should call the 'configure_denoising' method before using 'prepare'. Using default values.")
            self.configure_denoising(predict_noise=True, include_timestep=False, noise_intensity=0.1)
        
        
        if custom_transforms_train is None:
            train_transforms = []
            if augment:
                train_transforms.append(transforms.RandomHorizontalFlip())
                train_transforms.append(transforms.RandomRotation(10))
            train_transforms.append(transforms.Resize(target_size))
            train_transforms.append(transforms.ToTensor())
            train_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            train_transforms = custom_transforms_train
        
        if custom_transforms_val_test is None:
            val_test_transforms = [
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            val_test_transforms = custom_transforms_val_test

        train_transform = transforms.Compose(train_transforms)
        val_test_transform = transforms.Compose(val_test_transforms)

        train_dataset = CustomDataset(
            self.train_set,
            train_transform,
            self.predict_noise,
            self.include_timestep,
            self.scheduler
        )
        valid_dataset = CustomDataset(
            self.valid_set,
            val_test_transform,
            self.predict_noise,
            self.include_timestep,
            self.scheduler
        )
        test_dataset = CustomDataset(
            self.test_set,
            val_test_transform,
            self.predict_noise,
            self.include_timestep,
            self.scheduler
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader

    def plot_dataloader_samples(self, loader, num_samples=6, random_samples=True):
        """
        Display samples from a DataLoader in a subplot.

        Parameters:
        ------------
        loader : DataLoader
            Instance of DataLoader containing the data to display.
        num_samples : int
            Number of samples to display (total number of images in the subplot).
        random_samples : bool
            If True, randomly selects the samples to display. Otherwise, displays the first elements from the DataLoader.
        """
        import math

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        cols = 6
        rows = math.ceil(num_samples / (cols // 2))
        
        data = list(loader)
        if random_samples:
            indices = torch.randint(0, len(data[0][0]), (num_samples,))
        else:
            indices = torch.arange(0, min(num_samples, len(data[0][0])))
        
        _, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()

        def denormalize(img):
            img = img.permute(1, 2, 0)
            # Denormalize using ImageNet mean and std
            img = img * std + mean
            return img.clamp(0, 1).numpy()

        for i, idx in enumerate(indices):
            noisy_img = data[0][0][idx]
            target_img = data[0][1][idx]
            timestep = data[0][2][idx] if len(data[0]) > 2 else None

            noisy_img = denormalize(noisy_img)
            target_img = denormalize(target_img)

            axes[2 * i].imshow(noisy_img)
            axes[2 * i].axis('off')
            axes[2 * i].set_title("Noisy")

            axes[2 * i + 1].imshow(target_img)
            axes[2 * i + 1].axis('off')
            title = "Target"
            if timestep is not None:
                title += f"\nTimestep: {timestep.item()}"
            axes[2 * i + 1].set_title(title)

        for j in range(2 * len(indices), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
