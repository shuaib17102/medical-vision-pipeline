import os
from PIL import Image, ImageDraw

img_dir = "data/ISIC2018_Task1-2_Training_Input"
mask_dir = "data/ISIC2018_Task1_Training_GroundTruth"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

for i in range(32):
    # Make a fake pink skin picture with a brown circle in the middle
    img = Image.new("RGB", (256, 256), color=(230, 180, 170))
    draw_img = ImageDraw.Draw(img)
    draw_img.ellipse([80, 80, 176, 176], fill=(90, 50, 40))
    img.save(f"{img_dir}/ISIC_{i:07d}.jpg")

    # Make the matching black-and-white answer mask
    mask = Image.new("L", (256, 256), color=0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse([80, 80, 176, 176], fill=255)
    mask.save(f"{mask_dir}/ISIC_{i:07d}_segmentation.png")

print("SUCCESS! Created 32 practice images and masks!")