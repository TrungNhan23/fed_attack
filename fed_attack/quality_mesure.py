import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torchvision import datasets, transforms

# Load 1 ảnh từ CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
img, _ = cifar10[0]  # Chọn ảnh đầu tiên
img = img.permute(1, 2, 0).numpy()  # Chuyển về HWC, numpy, range [0,1]

# Tấn công: Gaussian noise
noise = np.random.normal(0, 0.1, img.shape)
img_gaussian = np.clip(img + noise, 0, 1)

# Tấn công: Salt & Pepper noise
img_sap = img.copy()
prob = 0.05
rnd = np.random.rand(*img.shape)
img_sap[rnd < (prob / 2)] = 0
img_sap[rnd > 1 - (prob / 2)] = 1

# Tấn công: Blur (dùng torch, nhưng hiển thị bằng numpy)
import torch.nn.functional as F
img_torch = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # BCHW
img_blur = F.avg_pool2d(img_torch, kernel_size=3, stride=1, padding=1)
img_blur = img_blur.squeeze(0).permute(1, 2, 0).numpy()

# Tính PSNR và SSIM
def calc_metrics(img1, img2):
    p = psnr(img1, img2, data_range=1.0)
    s = ssim(img1, img2, channel_axis=-1, data_range=1.0)
    return p, s

print("Gaussian Noise: PSNR =", *calc_metrics(img, img_gaussian))
print("Salt & Pepper: PSNR =", *calc_metrics(img, img_sap))
print("Blur: PSNR =", *calc_metrics(img, img_blur))

# Hiển thị ảnh
fig, axs = plt.subplots(1, 4, figsize=(12,3))
axs[0].imshow(img)
axs[0].set_title('Original')
axs[1].imshow(img_gaussian)
axs[1].set_title('Gaussian')
axs[2].imshow(img_sap)
axs[2].set_title('Salt & Pepper')
axs[3].imshow(img_blur)
axs[3].set_title('Blur')
for ax in axs:
    ax.axis('off')
plt.show()