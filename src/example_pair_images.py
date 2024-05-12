import matplotlib.pyplot as plt
from torchvision.io import read_image
import os 

current_folder = os.path.dirname(__file__)
image = read_image(f"{current_folder}/../data/PennFudanPed/PNGImages/FudanPed00046.png")
mask = read_image(f"{current_folder}/../data/PennFudanPed/PedMasks/FudanPed00046_mask.png")

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0))
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0))
plt.show()