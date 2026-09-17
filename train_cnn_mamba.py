import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from cnn_mamba_dehazer import CNNMambaDehazer


class DehazeDataset(Dataset):

    def __init__(self, hazy_folder, gt_folder):

        self.hazy_folder = hazy_folder
        self.gt_folder = gt_folder

        self.files = [
            f for f in os.listdir(hazy_folder)
            if f.lower().endswith(".png")
        ]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file = self.files[index]

        hazy = Image.open(
            os.path.join(self.hazy_folder, file)
        ).convert("RGB")

        gt = Image.open(
            os.path.join(self.gt_folder, file)
        ).convert("RGB")

        hazy = self.transform(hazy)
        gt = self.transform(gt)

        return hazy, gt


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


train_dataset = DehazeDataset(
    r"E:\DL project\UAVid_Dehazing\dataset\train\hazy",
    r"E:\DL project\UAVid_Dehazing\dataset\train\GT"
)

val_dataset = DehazeDataset(
    r"E:\DL project\UAVid_Dehazing\dataset\val\hazy",
    r"E:\DL project\UAVid_Dehazing\dataset\val\GT"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


model = CNNMambaDehazer().to(device)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

best_val_loss = float("inf")

os.makedirs(
    "models/cnn_mamba",
    exist_ok=True
)


for epoch in range(epochs):

    model.train()

    train_loss = 0

    for hazy, gt in train_loader:

        hazy = hazy.to(device)
        gt = gt.to(device)

        output = model(hazy)

        loss = loss_function(
            output,
            gt
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    model.eval()

    val_loss = 0

    with torch.no_grad():

        for hazy, gt in val_loader:

            hazy = hazy.to(device)
            gt = gt.to(device)

            output = model(hazy)

            loss = loss_function(
                output,
                gt
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Train MSE: {train_loss:.6f} "
        f"Val MSE: {val_loss:.6f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "models/cnn_mamba/cnn_mamba_dehazer.pth"
        )

        print("Best model saved!")


print()
print("Training completed!")

print(
    "Model saved at: "
    "models/cnn_mamba/cnn_mamba_dehazer.pth"
)