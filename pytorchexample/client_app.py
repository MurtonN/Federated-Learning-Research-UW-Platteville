"""pytorchexample: A Flower / PyTorch app."""

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# -------------------------------------------------------------
# 1. SETUP DATA METADATA (MEMORY-EFFICIENT)
# -------------------------------------------------------------
data_files = "Wednesday_workingHours_cleaned.csv"

# Read only headers and determine columns to save memory
sample_df = pd.read_csv(data_files, nrows=5)
sample_df.columns = sample_df.columns.str.strip()
target_column = "Label"  # Change to your target column name

features = [col for col in sample_df.columns if col != target_column and pd.api.types.is_numeric_dtype(sample_df[col])]
total_rows = sum(1 for _ in open(data_files)) - 1  # count rows excluding header

num_partitions = 5
partition_size = total_rows // num_partitions


# -------------------------------------------------------------
# 2. PYTORCH DATASET WRAPPER
# -------------------------------------------------------------
class PartitionDataset(Dataset):
    def __init__(self, partition_data, feature_cols, target_col):
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(partition_data[feature_cols])

        self.features = torch.tensor(x_scaled, dtype=torch.float32)
        self.labels = torch.tensor(partition_data[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Flower ClientApp
app = ClientApp()


def load_local_data(partition_id: int, batch_size: int):
    """Load *only* the specific chunk for this partition from disk to prevent OOM."""
    skip_rows = partition_id * partition_size + 1  # +1 for header
    nrows = partition_size if partition_id < num_partitions - 1 else None

    # Load only the rows belonging to this partition node
    partition_df = pd.read_csv(
        data_files,
        skiprows=range(1, skip_rows) if skip_rows > 1 else None,
        nrows=nrows,
        header=0
    )
    partition_df.columns = partition_df.columns.str.strip()

    partition_dataset = PartitionDataset(partition_df, features, target_column)

    # Split partition into train (80%) and validation (20%)
    train_size = int(0.8 * len(partition_dataset))
    val_size = len(partition_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(partition_dataset, [train_size, val_size])

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader


@app.train()
def train(msg: Message, context: Context) -> Message:
    config = msg.content.get("config", {}) if msg.has_content() else {}
    lr = config.get("lr", context.run_config.get("learning-rate", 0.01))

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_local_data(partition_id, batch_size)

    initial_params = {k: v.clone() for k, v in model.state_dict().items()}
    train_loss = train_fn(model, trainloader, context.run_config["local-epochs"], lr, device)

    state_dict = model.state_dict()

    # -------------------------------------------------------------
    # ADVANCED MALICIOUS BEHAVIOR (e.g., Client 0 as an Attacker)
    # -------------------------------------------------------------
    if partition_id == 0:
        print(f"Client {partition_id} executing intrusive malicious attack!")

        # Choose one of the following attack strategies:

        # Strategy A: Random Gaussian Noise Attack (Destroys model stability)
        # for key in state_dict:
        #     noise = torch.randn_like(state_dict[key]) * 10.0
        #     state_dict[key] = initial_params[key] + noise

        # Strategy B: Extreme Scaling / Gradient Bombing Attack
        for key in state_dict:
            delta = state_dict[key] - initial_params[key]
            # Magnify the malicious delta by a large factor to distort aggregation
            state_dict[key] = initial_params[key] + (50.0 * delta)

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num_examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    return Message(content=RecordDict({"arrays": model_record, "metrics": metric_record}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_local_data(partition_id, batch_size)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    return Message(content=RecordDict({"metrics": metric_record}), reply_to=msg)