#!/usr/bin/env python
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

class NoiseAdder:
    @staticmethod
    def add_gaussian(img_array, noise_level):
        noise = np.random.randn(*img_array.shape) * noise_level
        return img_array + noise

    @staticmethod
    def add_speckle(img_array, noise_level):
        noise = img_array * np.random.normal(0, noise_level, img_array.shape)
        return img_array + noise

    @staticmethod
    def add_drift(img_array, drift_strength):
        h, w = img_array.shape
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xv, yv = np.meshgrid(x, y)
        drift = drift_strength * (xv + yv**2)
        return img_array + drift

    @staticmethod
    def add_local_blur(img_array, patch_size=20, blur_prob=0.05, blur_sigma=1.0):
        blur_sigma = 10 * blur_sigma
        noisy = img_array.copy()
        h, w = img_array.shape
        num_patches = int((h * w) * blur_prob / (patch_size ** 2))
        for _ in range(num_patches):
            top = np.random.randint(0, h - patch_size)
            left = np.random.randint(0, w - patch_size)
            patch = noisy[top:top+patch_size, left:left+patch_size]
            blurred_patch = gaussian_filter(patch, sigma=blur_sigma)
            noisy[top:top+patch_size, left:left+patch_size] = blurred_patch
        return noisy

    @staticmethod
    def add_salt_pepper(img_array, prob):
        prob = 0.1 * prob
        noisy = img_array.copy()
        sp_mask = np.random.rand(*noisy.shape)
        noisy[sp_mask < prob/2] = 0.0
        noisy[sp_mask > 1 - prob/2] = 1.0
        return noisy

    @staticmethod
    def add_gamma(img_array):
        gamma = np.random.uniform(0.9, 1.1)
        return np.clip(img_array, 0, 1) ** gamma
    
    @staticmethod
    def add_gradient(img_array, c=0.3):
        assert img_array.ndim == 2, "Input must be a 2D image"
        sh = img_array.shape
        x, y = np.meshgrid(np.arange(0, sh[0]), np.arange(0, sh[1]), indexing='ij')

        c_eff = c * np.random.rand()
        angle = 2 * np.pi * np.random.rand()
        n = [np.cos(angle), np.sin(angle), 1]

        z = -(n[0] * x + n[1] * y) 
        z -= z.mean()
        z /= np.ptp(z)

        return img_array + z * c_eff * np.ptp(img_array)

    @staticmethod
    def add_noise(img_array, c=0.1, randomize_amplitude=True, normal_amplitude=True):
        assert img_array.ndim == 2, "Input must be a 2D image"
        sh = img_array.shape
        R = np.random.rand(*sh) - 0.5
        if randomize_amplitude:
            if normal_amplitude:
                amp = np.abs(np.random.normal(0, c, sh))
            else:
                amp = np.random.uniform(0.0, 1.0, sh)
        else:
            amp = np.full(sh, c)
        
        return img_array + R * amp * (img_array.max()-img_array.min())

    @staticmethod
    def add_cutout(img_array, n_holes=5):
        def get_random_eraser(input_img, p=0.2, s_l=0.001, s_h=0.01, r_1=0.1, r_2=1.0 / 0.1):
            """
            p : the probability that random erasing is performed
            s_l, s_h : minimum / maximum proportion of erased area against input image
            r_1, r_2 : minimum / maximum aspect ratio of erased area
            """

            sh = input_img.shape
            img_h, img_w = [sh[0], sh[1]]

            if np.random.uniform(0, 1) > p:
                return input_img

            while True:
                s = np.exp(np.random.uniform(np.log(s_l), np.log(s_h))) * img_h * img_w
                r = np.exp(np.random.uniform(np.log(r_1), np.log(r_2)))

                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, img_w)
                top = np.random.randint(0, img_h)

                if left + w <= img_w and top + h <= img_h:
                    break

            input_img[top : top + h, left : left + w] = 0.0

            return input_img

        img_array = img_array.copy() 
        for _ in range(n_holes):
            img_array = get_random_eraser(img_array)
        
        return img_array

def process_images(source_folder, target_folder, noise_type, noise_level=0.05):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path).convert('L')
            img_array = np.array(img).astype(np.float32) / 255.0

            if noise_type == 'gaussian':
                noised = NoiseAdder.add_gaussian(img_array, noise_level)
            elif noise_type == 'speckle':
                noised = NoiseAdder.add_speckle(img_array, noise_level)
            elif noise_type == 'drift':
                noised = NoiseAdder.add_drift(img_array, noise_level)
            elif noise_type == 'blur':
                noised = NoiseAdder.add_local_blur(img_array,  patch_size=40, blur_prob=0.5, blur_sigma=noise_level)
            elif noise_type == 'saltpepper':
                noised = NoiseAdder.add_salt_pepper(img_array, prob=noise_level)
            elif noise_type == 'gamma':
                noised = NoiseAdder.add_gamma(img_array)
            elif noise_type == 'gradient_previous':
                noised = NoiseAdder.add_gradient(img_array, c=0.3)
            elif noise_type == 'noise_previous':
                noised = NoiseAdder.add_noise(img_array, c=0.1, randomize_amplitude=True, normal_amplitude=True)
            elif noise_type == 'cutout_previous':
                noised = NoiseAdder.add_cutout(img_array, n_holes=5)
            elif noise_type == 'combined_previous':
                # Apply gradient_previous, then noise_previous, then cutout_previous in order
                noised = NoiseAdder.add_gradient(img_array, c=0.3)
                noised = NoiseAdder.add_noise(noised, c=0.1, randomize_amplitude=True, normal_amplitude=True)
                noised = NoiseAdder.add_cutout(noised, n_holes=5)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

            noised = np.clip(noised, 0, 1)
            noised = (noised * 255).astype(np.uint8)
            noised_img = Image.fromarray(noised)
            noised_img.save(os.path.join(target_folder, filename))

if __name__ == "__main__":
    # Example usage
    source_folder = '../../data/preEvaluate/realA'
    base_target_folder = '../../data/preEvaluate/noisedA_'
    #noise_types = ['gaussian', 'speckle', 'drift', 'blur', 'saltpepper', 'gamma']
    noise_types = ['gaussian', 'speckle', 'drift', 'blur', 'saltpepper']
    for noise_type in noise_types:
       print(f'Adding {noise_type} noise ...')
       for noise_level in [0.01, 0.05, 0.1, 0.2, 0.3, 0.35]:
           target_folder = f"{base_target_folder}{noise_type}_{noise_level}"
           process_images(source_folder, target_folder, noise_type, noise_level)

    # Previously used noise types
    noise_types = ['gradient_previous', 'noise_previous', 'cutout_previous', 'combined_previous']
    for noise_type in noise_types:
        print(f'Adding {noise_type} noise ...')
        for copy in [1, 2, 3, 4, 5, 6]:
            target_folder = f"{base_target_folder}{noise_type}_{copy}"
            process_images(source_folder, target_folder, noise_type)
