import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import math

from cnn_dehazer import CNNDehazer


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# Load model
model = CNNDehazer().to(device)

model.load_state_dict(
    torch.load(
        "models/cnn_dehazer.pth",
        map_location=device
    )
)

model.eval()


# Test folders
hazy_folder = (
    r"E:\DL project\UAVid_Dehazing"
    r"\dataset\test\hazy"
)

gt_folder = (
    r"E:\DL project\UAVid_Dehazing"
    r"\dataset\test\GT"
)

result_folder = (
    r"E:\DL project\UAVid_Dehazing"
    r"\results\cnn"
)


os.makedirs(
    result_folder,
    exist_ok=True
)


# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Simple SSIM
def calculate_ssim(img1, img2):

    C1 = 0.01 ** 2

    C2 = 0.03 ** 2

    mean1 = img1.mean()

    mean2 = img2.mean()

    variance1 = (
        (img1 - mean1) ** 2
    ).mean()

    variance2 = (
        (img2 - mean2) ** 2
    ).mean()

    covariance = (
        (img1 - mean1)
        * (img2 - mean2)
    ).mean()

    ssim = (
        (2 * mean1 * mean2 + C1)
        * (2 * covariance + C2)
    ) / (
        (mean1 ** 2 + mean2 ** 2 + C1)
        * (variance1 + variance2 + C2)
    )

    return ssim.item()


# Variables
total_mse = 0

total_psnr = 0

total_ssim = 0

count = 0


# Evaluation
with torch.no_grad():

    for file in os.listdir(hazy_folder):

        if not file.lower().endswith(".png"):
            continue


        hazy_path = os.path.join(
            hazy_folder,
            file
        )

        gt_path = os.path.join(
            gt_folder,
            file
        )


        hazy = Image.open(
            hazy_path
        ).convert("RGB")


        gt = Image.open(
            gt_path
        ).convert("RGB")


        hazy = transform(hazy)

        gt = transform(gt)


        hazy = hazy.unsqueeze(0).to(device)

        gt = gt.unsqueeze(0).to(device)


        # Model prediction
        output = model(hazy)


        # MSE
        mse = F.mse_loss(
            output,
            gt
        ).item()


        # PSNR
        if mse == 0:

            psnr = 100

        else:

            psnr = 10 * math.log10(
                1 / mse
            )


        # SSIM
        ssim = calculate_ssim(
            output[0],
            gt[0]
        )


        total_mse += mse

        total_psnr += psnr

        total_ssim += ssim

        count += 1


        # Save output
        output_image = output[0].cpu()

        output_image = transforms.ToPILImage()(
            output_image
        )

        output_image.save(
            os.path.join(
                result_folder,
                file
            )
        )


# Average results
average_mse = total_mse / count

average_psnr = total_psnr / count

average_ssim = total_ssim / count


# Display results
print()

print("==============================")

print("      BASIC CNN RESULTS")

print("==============================")

print(
    f"Test Images : {count}"
)

print(
    f"MSE         : {average_mse:.6f}"
)

print(
    f"PSNR        : {average_psnr:.2f} dB"
)

print(
    f"SSIM        : {average_ssim:.4f}"
)

print("==============================")

print()

print(
    "Dehazed images saved in:"
)

print(result_folder)