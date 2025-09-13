import sys
import os
import yaml
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ultralytics import YOLO
from utils.set_seed import set_seed

import wandb
# from wandb.integration.ultralytics import add_wandb_callback

set_seed(42)

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trainer(config=None):
    PROJECT = "ppe-yolo"
    NAME = "training"

    settings = wandb.Settings(disable_code=True, disable_git=True, _disable_stats=True)

    print("Initial config from sweep:", config)

    with wandb.init(project=PROJECT, job_type="train", settings=settings) as run:
        final_config = run.config
        print("HERE!!!", dict(final_config))
        
        model_name = final_config.get("model", "yolo11n.pt")
        data_path = final_config.get("dataset", "../data/data-ppe_v4.yaml")
        epochs = final_config.get("epochs", 100)
        batch_size = final_config.get("batch_size", 16)
        imgsz = final_config.get("imgsz", 640)
        
        print(f"Training with:")
        print(f"  Model: {model_name}")
        print(f"  Data: {data_path}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {imgsz}")

        model = YOLO(model_name)
        # add_wandb_callback(model, enable_model_checkpointing=True)

        model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=PROJECT,
            name=NAME,
        )

if __name__ == "__main__":
    trainer()