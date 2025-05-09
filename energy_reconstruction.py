# Import(s)
import os

from graphnet.constants import CONFIG_DIR  # Local path to graphnet/configs
from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.utilities.config import DatasetConfig, ModelConfig

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
    num_workers=63,
)

# Train model
model.fit(
    dataloaders["train"],
    dataloaders["test"],# Needs to be validation instead of test since we're also predicting test
    max_epochs=10,
)

# Predict on test set and return as pandas.DataFrame
results = model.predict_as_dataframe(
    dataloaders["test"],
    additional_attributes=model.target_labels + ["event_no"],
)

# Save predictions and model to file
print(results)
