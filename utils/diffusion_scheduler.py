import torch
import torch.nn.functional as F

class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, noise_intensity=None):
        """
        Diffusion scheduler for adding noise to images. It can also be used to add a certain amount of noise
        for denoising tasks (fixed or randomely sampled from a range).

        Parameters:
        -----------
        num_timesteps: int, optional
            Number of diffusion steps.
        noise_intensity: float or tuple(float, float), optional
            - If float, a fixed noise intensity will be used.
            - If tuple, a random noise intensity will be drawn from this range.
            - Default is None for diffusion tasks.
        """
        self.num_timesteps = num_timesteps
        self.noise_intensity = noise_intensity

        self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        
        # Pre-compute different terms for efficiency
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def add_noise(self, images, t):
        """
        Adds noise to images, supporting both diffusion and denoising modes.

        Parameters:
        -----------
        images: torch.Tensor
            Input images ([B, C, H, W]).
        t: torch.Tensor or None
            - Diffusion step indices for diffusion mode.
            - If None, adds random noise according to self.noise_intensity.

        Returns:
        --------
        noisy_images: torch.Tensor
            Images with added noise.
        noise: torch.Tensor
            Noise applied to the images.
        """
        if t is not None and not isinstance(t, torch.Tensor):
            raise ValueError("t must be a torch.Tensor or None.")

        noise = torch.randn_like(images)
        
        if t is not None: # Diffusion logic
            t = t.to(self.sqrt_alphas_cumprod.device)

            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
            
            noisy_images = sqrt_alphas_cumprod_t * images + sqrt_one_minus_alphas_cumprod_t * noise
            
        else: # Denoising logic
            if isinstance(self.noise_intensity, tuple):
                intensity = torch.rand(images.size(0), 1, 1, 1).to(images.device) * (self.noise_intensity[1] - self.noise_intensity[0]) + self.noise_intensity[0]
            elif isinstance(self.noise_intensity, float):
                intensity = self.noise_intensity
            else:
                intensity = 0.1
            
            noise = intensity * noise
            noisy_images = images + noise
        
        return noisy_images, noise

    def sample_timesteps(self, batch_size):
        """Sample timesteps uniformly for a batch of images"""
        return torch.randint(0, self.num_timesteps, (batch_size,))