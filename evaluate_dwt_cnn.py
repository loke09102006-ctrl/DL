import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import math

from dwt_cnn_dehazer import DWTCNNDehazer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


model = DWTCNNDehazer().to(device)

model.load_state_dict(
    torch.load(
        "models/dwt_cnn/dwt_cnn_dehazer.pth",
        map_location=device
    )
)

model.eval()


hazy_folder = r"E:\DL project\UAVid_Dehazing\dataset\test\hazy"
gt_folder = r"E:\DL project\UAVid_Dehazing\dataset\test\GT"

result_folder = r"E:\DL project\UAVid_Dehazing\results\dwt_cnn"

os.makedirs(result_folder, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def calculate_ssim(img1, img2):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mean1 = img1.mean()
    mean2 = img2.mean()

    variance1 = ((img1 - mean1) ** 2).mean()
    variance2 = ((img2 - mean2) ** 2).mean()

    covariance = (
        (img1 - mean1) *
        (img2 - mean2)
    ).mean()

    ssim = (
        (2 * mean1 * mean2 + C1) *
        (2 * covariance + C2)
    ) / (
        (mean1 ** 2 + mean2 ** 2 + C1) *
        (variance1 + variance2 + C2)
    )

    return ssim.item()


total_mse = 0
total_psnr = 0
total_ssim = 0

count = 0


with torch.no_grad():

    for file in os.listdir(hazy_folder):

        if not file.lower().endswith(".png"):
            continue

        hazy = Image.open(
            os.path.join(hazy_folder, file)
        ).convert("RGB")

        gt = Image.open(
            os.path.join(gt_folder, file)
        ).convert("RGB")

        hazy = transform(hazy)
        gt = transform(gt)

        hazy = hazy.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)

        output = model(hazy)

        mse = F.mse_loss(
            output,
            gt
        ).item()

        if mse == 0:
            psnr = 100
        else:
            psnr = 10 * math.log10(1 / mse)

        ssim = calculate_ssim(
            output[0],
            gt[0]
        )

        total_mse += mse
        total_psnr += psnr
        total_ssim += ssim

        count += 1

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


average_mse = total_mse / count
average_psnr = total_psnr / count
average_ssim = total_ssim / count


print()
print("==============================")
print("     DWT + CNN RESULTS")
print("==============================")

print(f"Test Images : {count}")
print(f"MSE         : {average_mse:.6f}")
print(f"PSNR        : {average_psnr:.6f} dB")
print(f"SSIM        : {average_ssim:.6f}")

print("==============================")

print()
print("Dehazed images saved in:")
print(result_folder)