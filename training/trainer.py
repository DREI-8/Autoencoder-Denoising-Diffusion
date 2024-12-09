import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.noise_scheduler import NoiseScheduler

class Trainer():
    def __init__(self, model, optimizer, critertion, device, type="denoise", noise_scheduler=None):
        """
        Trainer class for training and evaluating the model. It supports both denoising and diffusion tasks.

        Parameters
        ----------
        model: torch.nn.Module
            Model to be trained.
        optimizer: torch.optim.Optimizer
            Optimizer for the model.
        criterion: torch.nn.Module
            Loss function.
        device: torch.device
            Device to run the training on.
        type: str, optional
            Type of the task. Can be either "denoise" or "diffusion".
        noise_scheduler: NoiseScheduler, optional
            Noise scheduler for adding noise to images. If None, a default scheduler will be used.
        """
        if type not in ["denoise", "diffusion"]:
            raise ValueError("Type must be either 'denoise' or 'diffusion'.")
        self.model = model
        self.optimizer = optimizer
        self.critertion = critertion
        self.device = device
        self.diffusion = type == "diffusion"
        self.is_trained = False
        self.was_loaded = False
        
        if noise_scheduler is not None:
            self.noise_scheduler = noise_scheduler
        else:
            self.noise_scheduler = NoiseScheduler(num_timesteps=1000, noise_intensity=0.1)

        self.norm_mean = torch.tensor([0.485, 0.456, 0.406])
        self.norm_std = torch.tensor([0.229, 0.224, 0.225])

    def train_one_epoch(self, dataloader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader for the training data.

        Returns
        -------
        epoch_loss: float
            Average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0

        for _, batch in enumerate(dataloader):
            if len(batch) == 3:
                noisy_images, target_images, t = batch
                t = t.to(self.device)
            else:
                noisy_images, target_images = batch
                t = None

            noisy_images = noisy_images.to(self.device)
            target_images = target_images.to(self.device)
            self.optimizer.zero_grad()
            if t is not None:
                denoised_images = self.model(noisy_images, t)
            else:
                denoised_images = self.model(noisy_images)
            loss = self.critertion(denoised_images, target_images)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        return epoch_loss
    
    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation data.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader for the validation data.
        
        Returns
        -------
        epoch_loss: float
            Average loss for the epoch.
        """
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                if len(batch) == 3:
                    noisy_images, target_images, t = batch
                    t = t.to(self.device)
                else:
                    noisy_images, target_images = batch
                    t = None

                noisy_images = noisy_images.to(self.device)
                target_images = target_images.to(self.device)
                if t is not None:
                    denoised_images = self.model(noisy_images, t)
                else:
                    denoised_images = self.model(noisy_images)
                loss = self.critertion(denoised_images, target_images)
                epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        return epoch_loss
    
    def train(self, train_loader, valid_loader, num_epochs):
        """
        Trains the model for the specified number of epochs. The learning rate is updated using the CosineAnnealingLR.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            DataLoader for the training data.
        valid_loader: torch.utils.data.DataLoader
            DataLoader for the validation data.
        num_epochs: int
            Number of epochs to train the model.
        """
        self.train_losses = []
        self.valid_losses = []

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                              T_max=len(train_loader) * num_epochs,
                                                              last_epoch=-1,
                                                              eta_min=1e-9)
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'lr: {self.optimizer.param_groups[0]["lr"]}')

            train_loss = self.train_one_epoch(tqdm(train_loader, desc="Training"))
            valid_loss = self.evaluate(tqdm(valid_loader, desc="Validation"))

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            print(f'Train Loss: {train_loss:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}')
        
        self.is_trained = True
    
    def plot_metrics(self):
        """
        Plots the training and validation metrics.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.valid_losses, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def plot_examples(self, dataloader, num_examples=5, predict_type="noise"):
        """
        Plots the input and output images from the model. If the model is trained to predict noise,
        the predicted noise is subtracted from the input image to get the denoised image. If the model
        is trained to predict the image directly, the output of the model is the denoised image.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader for the data.
        num_examples: int, optional
            Number of examples to plot.
        predict_type: str, optional
            Type of the prediction. Can be either "noise" or "image".
        """
        if not self.is_trained and not self.was_loaded:
            raise ValueError("Model is not trained yet.")

        if self.diffusion:
            raise ValueError("Cannot plot examples for diffusion tasks. Please use the 'diffusion_inference' method.")
        
        if predict_type not in ["noise", "image"]:
            raise ValueError("Predict type must be either 'noise' or 'image'.")

        self.model.eval()
        with torch.no_grad():
            for i, (noisy_images, target_images) in enumerate(dataloader):
                noisy_images = noisy_images.to(self.device)

                if predict_type == "noise":
                    denoised_images = self.noise_scheduler.remove_noise(noisy_images, self.model, None)
                    target_images = self.noise_scheduler.restore_image_from_target_noise(noisy_images, target_images)
                else:
                    denoised_images = self.model(noisy_images)
                noisy_images = noisy_images.cpu()

                target_images = target_images.cpu()
                denoised_images = denoised_images.cpu()
                for j in range(num_examples):
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(self.denormalize(noisy_images[j]))
                    plt.title('Noisy Image')
                    plt.axis('off')
                    plt.subplot(1, 3, 2)
                    plt.imshow(self.denormalize(target_images[j]))
                    plt.title('Target Image')
                    plt.axis('off')
                    plt.subplot(1, 3, 3)
                    plt.imshow(self.denormalize(denoised_images[j]))
                    plt.title('Denoised Image')
                    plt.axis('off')
                    plt.show()
                    if j == num_examples - 1:
                        break
                if i == 0:
                    break
    
    def diffusion_inference(self, batch_size=1, img_size=64, channels=3, intermediate_steps_to_show=6):
        
        generated_images = self.noise_scheduler.generate_image(self.model, batch_size, img_size, channels, intermediate_steps_to_show)
        generated_images = generated_images.cpu()
        
        _, axes = plt.subplots(batch_size, intermediate_steps_to_show, figsize=(15, 3*batch_size))
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            for j in range(intermediate_steps_to_show):
                axes[i, j].imshow(self.denormalize(generated_images[j][i]))
                axes[i, j].axis('off')
                if i == 0:
                    if j == 0:
                        axes[i, j].set_title('Initial noise')
                    elif j == intermediate_steps_to_show - 1:
                        axes[i, j].set_title('Final image')
                    else:
                        exponent = 2.5
                        step_percent = int(100 * (1 - pow(1 - j / (intermediate_steps_to_show - 1), exponent)))
                        axes[i, j].set_title(f'Step {step_percent}%')
        
        plt.tight_layout()
        plt.show()
    
    def denormalize(self, img):
        """
        Denormalizes the image according to the mean and standard deviation.
        """
        img = img.permute(1, 2, 0)
        img = img * self.norm_std + self.norm_mean
        return img.clamp(0, 1).numpy()
    
    def save_model(self, path):
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path: str
            Path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path: str
            Path to load the model.
        """
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.was_loaded = True
