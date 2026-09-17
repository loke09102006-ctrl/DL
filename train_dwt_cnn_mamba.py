import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image

import os

from dwt_cnn_mamba_dehazer import DWTCNNMambaDehazer


# ==========================================
# DATASET
# ==========================================

class DehazeDataset(Dataset):

    def __init__(self, hazy_folder, gt_folder):

        self.hazy_folder = hazy_folder

        self.gt_folder = gt_folder

        self.files = [
            f
            for f in os.listdir(hazy_folder)
            if f.lower().endswith(".png")
        ]

        self.transform = transforms.Compose([

            transforms.Resize(
                (256, 256)
            ),

            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        file = self.files[index]

        hazy_path = os.path.join(
            self.hazy_folder,
            file
        )

        gt_path = os.path.join(
            self.gt_folder,
            file
        )

        hazy = Image.open(
            hazy_path
        ).convert("RGB")

        gt = Image.open(
            gt_path
        ).convert("RGB")

        hazy = self.transform(hazy)

        gt = self.transform(gt)

        return hazy, gt


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==========================================
# DATA
# ==========================================

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


print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


# ==========================================
# MODEL
# ==========================================

model = DWTCNNMambaDehazer().to(device)


# ==========================================
# LOSS
# ==========================================

loss_function = nn.MSELoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001
)


# ==========================================
# TRAINING SETTINGS
# ==========================================

epochs = 20

best_val_loss = float("inf")


# ==========================================
# MODEL DIRECTORY
# ==========================================

os.makedirs(
    "models/dwt_cnn_mamba",
    exist_ok=True
)


# ==========================================
# TRAINING
# ==========================================

for epoch in range(epochs):

    model.train()

    train_loss = 0


    for hazy, gt in train_loader:

        hazy = hazy.to(device)

        gt = gt.to(device)


        # Forward

        output = model(hazy)


        # Loss

        loss = loss_function(
            output,
            gt
        )


        # Backpropagation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        train_loss += loss.item()


    train_loss /= len(train_loader)


    # ======================================
    # VALIDATION
    # ======================================

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


    # ======================================
    # PRINT RESULTS
    # ======================================

    print(

        f"Epoch {epoch + 1}/{epochs} "

        f"Train MSE: {train_loss:.6f} "

        f"Val MSE: {val_loss:.6f}"

    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        torch.save(

            model.state_dict(),

            "models/dwt_cnn_mamba/"
            "dwt_cnn_mamba_dehazer.pth"

        )


        print("Best model saved!")


# ==========================================
# COMPLETE
# ==========================================

print()

print(
    "================================"
)

print(
    "Training completed!"
)

print(
    "================================"
)

print(
    "Model saved at:"
)

print(
    "models/dwt_cnn_mamba/"
    "dwt_cnn_mamba_dehazer.pth"
)