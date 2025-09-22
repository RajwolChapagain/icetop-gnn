# Import(s)
import os

from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.utilities.config import DatasetConfig, ModelConfig
from graphnet.data.dataset import Dataset
from graphnet.training.callbacks import ProgressBar, EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# Configuration
dataset_config_path = f"configs/dataset_config.yml"
model_config_path = f"configs/model_config.yml"

# Build model
model_config = ModelConfig.load(model_config_path)
model = Model.from_config(model_config, trust=True)

# Construct dataloaders
dataset_config = DatasetConfig.load(dataset_config_path)
dataloaders = DataLoader.from_dataset_config(
    dataset_config,
    batch_size=128,
    num_workers=24,
)

# Callbacks
callbacks = [
    ProgressBar(),
    EarlyStopping(monitor="val_loss", patience=15, mode="min"),  # stop if no progress
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
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)
results.to_csv(f"{outdir}/results.csv")
model.save_state_dict(f"{outdir}/state_dict.pth")
model.save(f"{outdir}/model.pth")
