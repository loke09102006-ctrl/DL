import os
import math
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from cnn_mamba_dehazer import CNNMambaDehazer


# -------------------- SETTINGS --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hazy_folder = r"E:\DL project\UAVid_Dehazing\dataset\test\hazy"
gt_folder = r"E:\DL project\UAVid_Dehazing\dataset\test\GT"
result_folder = r"E:\DL project\UAVid_Dehazing\results\cnn_mamba"
model_path = r"E:\DL project\UAVid_Dehazing\models\cnn_mamba\cnn_mamba_dehazer.pth"

os.makedirs(result_folder, exist_ok=True)

to_tensor = transforms.ToTensor()

print("Using device:", device)


# -------------------- LOAD MODEL --------------------
model = CNNMambaDehazer().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("CNN + Mamba model loaded successfully.")


# -------------------- DEHAZE LARGE IMAGE --------------------
def dehaze_large_image(image, tile_size=256):
    original_width, original_height = image.size

    image_tensor = to_tensor(image).unsqueeze(0).to(device)
    output_image = torch.zeros_like(image_tensor)

    for y in range(0, original_height, tile_size):
        for x in range(0, original_width, tile_size):

            tile = image_tensor[:, :, y:min(y + tile_size, original_height),
                                x:min(x + tile_size, original_width)]

            tile_height = tile.shape[2]
            tile_width = tile.shape[3]

            pad_bottom = tile_size - tile_height
            pad_right = tile_size - tile_width

            # Pad edge tiles to 256x256
            if pad_bottom > 0 or pad_right > 0:
                tile = F.pad(
                    tile,
                    (0, pad_right, 0, pad_bottom),
                    mode="replicate"
                )

            # Run CNN + Mamba model
            restored_tile = model(tile)

            # Remove added padding
            restored_tile = restored_tile[:, :, :tile_height, :tile_width]

            # Put tile back into original position
            output_image[
                :,
                :,
                y:y + tile_height,
                x:x + tile_width
            ] = restored_tile

    output_image = output_image.squeeze(0).clamp(0, 1)

    return output_image


# -------------------- PSNR --------------------
def calculate_psnr(output, target):
    mse = F.mse_loss(output, target).item()

    if mse == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse)


# -------------------- SIMPLE SSIM --------------------
def calculate_ssim(output, target):
    output_mean = output.mean()
    target_mean = target.mean()

    output_var = ((output - output_mean) ** 2).mean()
    target_var = ((target - target_mean) ** 2).mean()

    covariance = ((output - output_mean) * (target - target_mean)).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = (
        (2 * output_mean * target_mean + c1) *
        (2 * covariance + c2)
    ) / (
        (output_mean ** 2 + target_mean ** 2 + c1) *
        (output_var + target_var + c2)
    )

    return ssim.item()


# -------------------- TESTING --------------------
total_psnr = 0
total_ssim = 0
count = 0

with torch.no_grad():

    for file in sorted(os.listdir(hazy_folder)):

        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        hazy_path = os.path.join(hazy_folder, file)
        gt_path = os.path.join(gt_folder, file)

        if not os.path.exists(gt_path):
            print("GT not found:", file)
            continue

        hazy_image = Image.open(hazy_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")

        print("\nProcessing:", file)
        print("Original size:", hazy_image.size)

        # Dehaze while preserving original resolution
        output = dehaze_large_image(hazy_image, tile_size=256)

        # Load GT without resizing
        gt = to_tensor(gt_image).to(device)

        # Check dimensions
        if output.shape != gt.shape:
            print("Size mismatch:", file)
            print("Output:", output.shape)
            print("GT:", gt.shape)
            continue

        # Calculate metrics
        psnr = calculate_psnr(output, gt)
        ssim = calculate_ssim(output, gt)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

        # Save result
        output_image = transforms.ToPILImage()(output.cpu())
        output_path = os.path.join(result_folder, file)
        output_image.save(output_path)

        print("Output size:", output_image.size)
        print(f"PSNR: {psnr:.4f}")
        print(f"SSIM: {ssim:.4f}")
        print("Saved:", output_path)


# -------------------- AVERAGE RESULTS --------------------
if count > 0:
    print("\n--------------------------------")
    print("CNN + Mamba Evaluation Complete")
    print("--------------------------------")
    print(f"Images processed: {count}")
    print(f"Average PSNR: {total_psnr / count:.4f}")
    print(f"Average SSIM: {total_ssim / count:.4f}")
else:
    print("\nNo images were processed.")