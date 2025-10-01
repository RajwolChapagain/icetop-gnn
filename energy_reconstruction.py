# Import(s)
import os
import argparse
import torch
from torch.utils.data import random_split
from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.utilities.config import DatasetConfig, ModelConfig
from graphnet.data.dataset import Dataset
from graphnet.training.callbacks import ProgressBar, EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--model-config", "-mc",
    type=str,
    required=True,
    help="Model config filename under configs directory"
)
parser.add_argument(
    "--dataset-config", "-dc",
    type=str,
    required=True,
    help="Dataset config filename under configs directory"
)
parser.add_argument(
    "--selection", "-s",
    type=str,
    required=True,
    help="Selection to use as defined in dataset config"
)
parser.add_argument(
    "--outdir", "-o",
    type=str,
    required=True,
    help="Directory under outputs directory where model results are saved"
)

args = parser.parse_args()

PROJECT_ROOT_DIR = '/home/rchapagain/icetop-gnn'
# Configuration
dataset_config_path = f"{PROJECT_ROOT_DIR}/configs/{args.dataset_config}.yml"
model_config_path = f"{PROJECT_ROOT_DIR}/configs/{args.model_config}.yml"

# Build model
model_config = ModelConfig.load(model_config_path)
model = Model.from_config(model_config, trust=True)

# Construct dataloaders
dataset = Dataset.from_config(dataset_config_path)[args.selection]
total_len = len(dataset)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset  = random_split(dataset, [train_len, val_len, test_len], generator=generator)

dataloaders = {
    "train": DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=24),
    "validation": DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=24),
    "test": DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=24),
}

# Callbacks
callbacks = [
    ProgressBar(),
    EarlyStopping(monitor="val_loss", patience=10, mode="min"),  # stop if no progress
    ModelCheckpoint(dirpath="checkpoints", save_top_k=1, monitor="val_loss", mode="min"),
]

# Train model
model.fit(
    dataloaders["train"],
    dataloaders["validation"],
    max_epochs=200,
    gpus=[0],
    callbacks=callbacks,
)

# Predict on test set and return as pandas.DataFrame
results = model.predict_as_dataframe(
    dataloaders["test"],
    additional_attributes=model.target_labels + ["event_no"],
    gpus=[0],
)

# Save predictions and model to file
print(results)
outdir = f"outputs/{args.outdir}"
os.makedirs(outdir, exist_ok=True)
results.to_csv(f"{outdir}/results.csv")
model.save_state_dict(f"{outdir}/state_dict.pth")
model.save(f"{outdir}/model.pth")
