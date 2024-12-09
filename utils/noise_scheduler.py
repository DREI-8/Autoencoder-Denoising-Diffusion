import torch
import torch.nn.functional as F
from tqdm import tqdm

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, noise_intensity=None, device = 'cuda'):
        """
        Diffusion scheduler for adding noise to images. It can also be used to add a certain amount of noise
        for denoising tasks (fixed or randomely sampled from a range).

        Parameters
        ----------
        num_timesteps: int, optional
            Number of diffusion steps.
        noise_intensity: float or tuple(float, float), optional
            - If float, a fixed noise intensity will be used.
            - If tuple, a random noise intensity will be drawn from this range.
            - Default is None for diffusion tasks.
        device: str, optional
            Device to run the scheduler on.
        """
        self.device = device
        self.num_timesteps = num_timesteps
        self.noise_intensity = noise_intensity

        # The values found in the paper DDPM (Ho et al., 2020) are used here
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps).to(self.device)  # Proportion of noise to add at each timestep
        
        # Pre-compute different terms for efficiency
        self.alphas = (1. - self.betas).to(self.device)                           # Proportion of image to keep at each timestep
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)   # Proportion of the original image that remains after a given timestep
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_1_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)

    def add_noise(self, images, t):
        """
        Adds noise to images, supporting both diffusion and denoising modes.

        Parameters
        ----------
        images: torch.Tensor
            Input images ([B, C, H, W]).
        t: torch.Tensor or None
            - Diffusion step indices for diffusion mode.
            - If None, adds random noise according to self.noise_intensity.

        Returns
        -------
        noisy_images: torch.Tensor
            Images with added noise.
        noise: torch.Tensor
            Noise applied to the images.
        """
        if t is not None and not isinstance(t, torch.Tensor):
            raise ValueError("t must be a torch.Tensor or None.")

        images = images.to(self.device)
        noise = torch.randn_like(images)
        
        if t is not None: # Diffusion logic
            t = t.to(self.sqrt_alphas_cumprod.device)

            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_1_minus_alphas_cumprod_t = self.sqrt_1_minus_alphas_cumprod[t]
            
            # Expand dimensions for broadcasting
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
            sqrt_1_minus_alphas_cumprod_t = sqrt_1_minus_alphas_cumprod_t[:, None, None, None]
            
            noisy_images = sqrt_alphas_cumprod_t * images + sqrt_1_minus_alphas_cumprod_t * noise
            
        else: # Denoising logic
            if isinstance(self.noise_intensity, tuple):
                intensity = torch.rand(images.size(0), 1, 1, 1).to(images.device) * (self.noise_intensity[1] - self.noise_intensity[0]) + self.noise_intensity[0]
            elif isinstance(self.noise_intensity, float):
                intensity = self.noise_intensity
            else:
                intensity = 0.1
            
            noisy_images = images * (1-intensity) + intensity * noise
        
        return noisy_images, noise
    
    def remove_noise(self, noisy_images, model, t):
        """
        Removes noise from images, supporting both diffusion and denoising modes.
        If t is None, denoising mode is used, and the completely denoised images are returned.
        If t is not None, diffusion mode is used, and the images are denoised at the given timestep.

        Parameters
        ----------
        noisy_images: torch.Tensor
            Noisy images ([B, C, H, W]).
        t: torch.Tensor or None
            - Diffusion step indices for diffusion mode.
            - If None, removes noise according to self.noise_intensity.

        Returns
        -------
        denoised_images: torch.Tensor
            Images with removed noise.
        """
        if t is not None and not isinstance(t, torch.Tensor):
            raise ValueError("t must be a torch.Tensor or None.")
        
        if t is not None:
            predicted_noise = model(noisy_images, t)

            alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)

            mask = (t > 0).reshape(-1, 1, 1, 1)

            noise = torch.randn_like(noisy_images).to(noisy_images.device)

            denoised_t_greater_0 = (1. / torch.sqrt(alpha_t)) * (noisy_images - (beta_t / (torch.sqrt(1. - self.alphas_cumprod[t]).reshape(-1, 1, 1, 1))) \
                            * predicted_noise) + torch.sqrt(beta_t) * noise
            
            denoised_t_0 = (noisy_images - torch.sqrt(1. - self.alphas_cumprod[t]).reshape(-1, 1, 1, 1) * predicted_noise) / \
                    torch.sqrt(self.alphas_cumprod[t]).reshape(-1, 1, 1, 1)
            
            denoised_images = torch.where(mask, denoised_t_greater_0, denoised_t_0)
            
        else:
            predicted_noise = model(noisy_images)
            denoised_images = 1/(1-self.noise_intensity) * (noisy_images - self.noise_intensity * predicted_noise)
        
        return denoised_images
    
    def restore_image_from_target_noise(self, noisy_images, target_noise):
        """
        Restores the original image from noisy images and target noise.
        Only used for denoising tasks, and not for diffusion.

        Parameters
        ----------
        noisy_images: torch.Tensor
            Noisy images ([B, C, H, W]).
        target_noise: torch.Tensor
            Target noise ([B, C, H, W]).

        Returns
        -------
        restored_images: torch.Tensor
            Restored images.
        """
        return 1/(1-self.noise_intensity) * (noisy_images - self.noise_intensity * target_noise)

    def generate_image(self, model, batch_size=1, img_size=64, channels=3, intermediate_steps_to_show=1):
        """
        Generates an image by sampling noise and denoising it iteratively. 
        If intermediate_steps_to_show is greater than 1, intermediate images are also returned,
        with more intermediate steps shown towards the end of the generation process.

        Parameters
        ----------
        model: torch.nn.Module
            Model used for denoising.
        batch_size: int, optional
            Batch size for generating images.
        img_size: int, optional
            Image size.
        channels: int, optional
            Number of image channels.
        device: str, optional
            Device to run the model on.
        intermediate_steps_to_show: int, optional
            Number of intermediate steps to show. If 1, only the final image is returned. If 2, the initial noise and final image are returned.
            If greater than 2, the initial noise, final image, and a number of intermediate images are returned.

        Returns
        -------
        generated_images: torch.Tensor
            Generated images ([intermediate_steps_to_show, B, C, H, W]).
        """
        generated_image = torch.randn(batch_size, channels, img_size, img_size).to(self.device)
        images_to_show = [generated_image.clone()]

        steps = list(reversed(range(self.num_timesteps)))

        if intermediate_steps_to_show > 1:
            exponent = 2.5
            normalized_indices = torch.linspace(0, 1, intermediate_steps_to_show - 1)
            normalized_indices = 1 - torch.pow(1 - normalized_indices, exponent)
            step_indices = (normalized_indices * (len(steps) - 1)).long().tolist()
            step_indices = sorted(list(set(step_indices)))
        else:
            step_indices = []

        for i, t in enumerate(tqdm(steps, desc="Generating image")):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                generated_image = self.remove_noise(generated_image, model, t_batch)
            if i in step_indices:
                images_to_show.append(generated_image.clone())

        images_to_show.append(generated_image.clone())
        return torch.stack(images_to_show)

    def sample_timesteps(self, batch_size):
        """Sample timesteps uniformly for a batch of images"""
        return torch.randint(0, self.num_timesteps, (batch_size,))