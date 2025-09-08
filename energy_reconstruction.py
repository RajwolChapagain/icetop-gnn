# Import(s)
import os

from graphnet.constants import CONFIG_DIR  # Local path to graphnet/configs
from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.utilities.config import DatasetConfig, ModelConfig
from graphnet.data.dataset import Dataset

# Configuration
dataset_config_path = f"test.yml"
model_config_path = f"{CONFIG_DIR}/models/icetop_energy_reconstruction_model.yml"

# Build model
model_config = ModelConfig.load(model_config_path)
model = Model.from_config(model_config, trust=True)

# Construct dataloaders
dataset_config = DatasetConfig.load(dataset_config_path)
dataloaders = DataLoader.from_dataset_config(
    dataset_config,
    batch_size=16,
    num_workers=24,
)

# Train model
model.fit(
    dataloaders["train"],
    max_epochs=50,
    gpus=[0],
)

# Predict on test set and return as pandas.DataFrame
results = model.predict_as_dataframe(
    dataloaders["test"],
    additional_attributes=model.target_labels + ["event_no"],
    gpus=[0],
)

# Save predictions and model to file
print(results)
outdir = "predictions"
os.makedirs(outdir, exist_ok=True)
results.to_csv(f"{outdir}/results.csv")
model.save_state_dict(f"{outdir}/state_dict.pth")
model.save(f"{outdir}/model.pth")
