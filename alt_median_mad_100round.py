# ALT ENTROPY-SHIFT BASELINE — 100 ROUNDS
# Same 20-client IID/Non-IID PlantVillage partition and SimpleCNN as earlier experiments.
# Each client calculates Shannon prediction entropy on its own held-out validation set
# before and after one local training epoch, then reports absolute shift delta_h.
# This is a DIAGNOSTIC baseline: all 20 valid client updates enter FedAvg.
# No Median+MAD or EMA threshold is applied in this file.
# Accuracy and per-client entropy-shift measurements are logged each round.


import csv
import random
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from torch.utils.data import (
    DataLoader,
    Subset,
)

from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import (
    SimpleCNN,
    IMAGE_SIZE,
    NUM_CLASSES,
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_DIR = "color"

PARTITION_FILE = Path(
    "mixed_10iid_10noniid_partitions.npz"
)

NUM_CLIENTS = 20
NUM_IID_CLIENTS = 10

# Full 100-round Median + MAD experiment
NUM_ROUNDS = 100
MAD_MULTIPLIER = 2.0

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

SEED = 42


# Output files
ACCURACY_CSV = Path(
    "alt_median_mad_100round_accuracy.csv"
)

ENTROPY_CSV = Path(
    "alt_median_mad_100round_clients.csv"
)

FINAL_MODEL = Path(
    "alt_median_mad_100round_final_model.pt"
)


# ============================================================
# 2. RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"\nUsing device: {DEVICE}",
    flush=True,
)


# ============================================================
# 4. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
])


# ============================================================
# 5. LOAD FULL DATASET
# ============================================================

full_dataset = ImageFolder(
    DATA_DIR,
    transform=transform,
)


if len(full_dataset) == 0:

    raise RuntimeError(
        "Dataset is empty."
    )


if len(full_dataset.classes) != NUM_CLASSES:

    raise RuntimeError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found "
        f"{len(full_dataset.classes)}."
    )


print(
    f"Total images: {len(full_dataset)}",
    flush=True,
)

print(
    f"Total classes: "
    f"{len(full_dataset.classes)}",
    flush=True,
)


# ============================================================
# 6. LOAD SAVED 20-CLIENT PARTITION
# ============================================================

if not PARTITION_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{PARTITION_FILE}\n\n"
        f"Make sure the saved partition "
        f"file is in this folder."
    )


partitions = np.load(
    PARTITION_FILE,
    allow_pickle=True,
)


# Check dataset size
saved_dataset_size = int(
    partitions["dataset_size"]
)

if saved_dataset_size != len(full_dataset):

    raise RuntimeError(
        "Dataset size does not match "
        "the saved partition."
    )


# ============================================================
# 7. BUILD CLIENT DATASETS
# ============================================================

client_train_datasets = []
client_val_datasets = []


for cid in range(NUM_CLIENTS):

    train_indices = partitions[
        f"train_{cid}"
    ]

    val_indices = partitions[
        f"val_{cid}"
    ]

    client_train_datasets.append(
        Subset(
            full_dataset,
            train_indices,
        )
    )

    client_val_datasets.append(
        Subset(
            full_dataset,
            val_indices,
        )
    )


# ============================================================
# 8. GLOBAL TEST DATA
# ============================================================

global_test_indices = partitions[
    "global_test_indices"
]


global_test_dataset = Subset(
    full_dataset,
    global_test_indices,
)


global_test_loader = DataLoader(
    global_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


print(
    f"Global test images: "
    f"{len(global_test_dataset)}",
    flush=True,
)


# ============================================================
# 9. SHOW CLIENT INFORMATION
# ============================================================

print(
    "\n========================================"
)

print(
    " CLIENT DATA"
)

print(
    "========================================"
)


for cid in range(NUM_CLIENTS):

    client_type = (
        "IID"
        if cid < NUM_IID_CLIENTS
        else "Non-IID"
    )

    print(
        f"Client {cid:02d} | "
        f"{client_type:7s} | "
        f"train="
        f"{len(client_train_datasets[cid])} | "
        f"val="
        f"{len(client_val_datasets[cid])}"
    )


# ============================================================
# 10. MODEL PARAMETER FUNCTIONS
# ============================================================

def get_model_parameters(model):

    return [
        value.detach().cpu().numpy()
        for value
        in model.state_dict().values()
    ]


def set_model_parameters(
    model,
    parameters,
):

    state_dict = model.state_dict()

    new_state_dict = {}

    for key, value in zip(
        state_dict.keys(),
        parameters,
    ):

        new_state_dict[key] = (
            torch.tensor(value)
        )

    model.load_state_dict(
        new_state_dict,
        strict=True,
    )


# ============================================================
# 11. LOCAL TRAINING
# ============================================================

def train_locally(
    model,
    dataloader,
):

    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )


    total_loss = 0.0
    total_samples = 0


    for _ in range(LOCAL_EPOCHS):

        for images, targets in dataloader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

            loss.backward()

            optimizer.step()


            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size


    if total_samples == 0:
        return 0.0


    return (
        total_loss
        / total_samples
    )


# ============================================================
# 12. PREDICTION ENTROPY
#
# H = - sum(p * log(p))
#
# IMPORTANT:
# Entropy is measured AFTER local training
# using the client's private validation data.
# ============================================================

def compute_prediction_entropy(
    model,
    dataloader,
):

    model.eval()

    entropy_values = []


    with torch.no_grad():

        for images, _ in dataloader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = F.softmax(
                outputs,
                dim=1,
            )


            entropy = -torch.sum(
                probabilities
                * torch.log(
                    probabilities + 1e-12
                ),
                dim=1,
            )


            entropy_values.extend(
                entropy
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


    if len(entropy_values) == 0:

        return 0.0


    return float(
        np.mean(entropy_values)
    )


# ============================================================
# 13. GLOBAL TEST ACCURACY
# ============================================================

def evaluate_accuracy(
    model,
    dataloader,
):

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for images, targets in dataloader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)


    if total == 0:
        return 0.0


    return correct / total


# ============================================================
# 14. FLOWER CLIENT
# ============================================================

class FlowerClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        client_id,
    ):

        self.client_id = client_id


        self.model = SimpleCNN(
            num_classes=NUM_CLASSES
        ).to(DEVICE)


        self.train_loader = DataLoader(
            client_train_datasets[
                client_id
            ],
            batch_size=BATCH_SIZE,
            shuffle=True,
        )


        self.val_loader = DataLoader(
            client_val_datasets[
                client_id
            ],
            batch_size=BATCH_SIZE,
            shuffle=False,
        )


    # --------------------------------------------------------
    # GET PARAMETERS
    # --------------------------------------------------------

    def get_parameters(
        self,
        config,
    ):

        return get_model_parameters(
            self.model
        )


    # --------------------------------------------------------
    # LOCAL TRAINING
    # --------------------------------------------------------

    def fit(
        self,
        parameters,
        config,
    ):

        # Every round the client receives
        # the SAME current global model.
        set_model_parameters(
            self.model,
            parameters,
        )


        # Baseline entropy BEFORE local training (same private validation set).
        h_start = compute_prediction_entropy(self.model, self.val_loader)

        # Train locally
        local_loss = train_locally(
            self.model,
            self.train_loader,
        )


        # Prediction entropy AFTER training
        h_end = compute_prediction_entropy(
            self.model,
            self.val_loader,
        )


        client_type = (
            "IID"
            if self.client_id
            < NUM_IID_CLIENTS
            else "Non-IID"
        )


        print(
            f"Client "
            f"{self.client_id:02d} | "
            f"{client_type:7s} | "
            f"loss={local_loss:.4f} | "
            f"H_start={h_start:.6f} | H_end={h_end:.6f} | "
            f"delta_H={abs(h_end - h_start):.6f}",
            flush=True,
        )


        # Diagnostic only: send the update and entropy statistics.
        # The server aggregates ALL 20 updates without entropy gating.
        metrics = {

            "client_id":
                int(self.client_id),

            "entropy":
                float(h_end),

            "h_start": float(h_start),
            "h_end": float(h_end),
            "delta_h": float(abs(h_end - h_start)),

            "client_type":
                client_type,

            "local_loss":
                float(local_loss),
        }


        return (

            get_model_parameters(
                self.model
            ),

            len(
                self.train_loader.dataset
            ),

            metrics,
        )


    # --------------------------------------------------------
    # CLIENT EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        parameters,
        config,
    ):

        set_model_parameters(
            self.model,
            parameters,
        )

        return (
            0.0,
            len(
                self.val_loader.dataset
            ),
            {},
        )


# ============================================================
# 15. CLIENT FUNCTION
# ============================================================

def client_fn(cid):

    return FlowerClient(
        int(cid)
    ).to_client()



# ============================================================
# CSV output: 100-round Median + MAD client selection
# ============================================================
with open(ACCURACY_CSV, "w", newline="") as f:
    csv.writer(f).writerow(
        ["round", "accuracy", "received_clients", "aggregated_clients",
         "median_delta_h", "mad_delta_h", "threshold"]
    )

with open(ENTROPY_CSV, "w", newline="") as f:
    csv.writer(f).writerow(
        ["round", "client_id", "client_type", "h_start", "h_end",
         "delta_h", "local_loss", "threshold", "status"]
    )

accuracy_history = []
final_parameters = None


class AltMedianMADFedAvg(fl.server.strategy.FedAvg):
    """All clients train; server aggregates only updates with ΔH <= median + 2*MAD."""

    def aggregate_fit(self, server_round, results, failures):
        global final_parameters

        if failures or len(results) != NUM_CLIENTS:
            raise RuntimeError(
                f"Round {server_round}: expected {NUM_CLIENTS} successful clients; "
                f"got {len(results)} results and {len(failures)} failures."
            )

        records = []
        for client_proxy, fit_res in results:
            m = fit_res.metrics
            records.append({
                "client_id": int(m["client_id"]),
                "client_type": str(m["client_type"]),
                "h_start": float(m["h_start"]),
                "h_end": float(m["h_end"]),
                "delta_h": float(m["delta_h"]),
                "local_loss": float(m["local_loss"]),
                "result": (client_proxy, fit_res),
            })

        ids = [r["client_id"] for r in records]
        if len(set(ids)) != NUM_CLIENTS or set(ids) != set(range(NUM_CLIENTS)):
            raise RuntimeError(f"Missing or duplicate client IDs: {ids}")

        shifts = np.array([r["delta_h"] for r in records], dtype=float)
        if not np.all(np.isfinite(shifts)):
            raise RuntimeError(f"Round {server_round}: nonfinite entropy shifts.")

        median_shift = float(np.median(shifts))
        mad_shift = float(np.median(np.abs(shifts - median_shift)))
        threshold = median_shift + MAD_MULTIPLIER * mad_shift

        selected = [r for r in records if r["delta_h"] <= threshold]
        # Numerical ties / a zero MAD can leave too few accepted clients.
        # Ensure at least one update so FedAvg can safely aggregate.
        if not selected:
            selected = [min(records, key=lambda r: r["delta_h"])]
        selected_ids = {r["client_id"] for r in selected}

        with open(ENTROPY_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            for r in sorted(records, key=lambda x: x["client_id"]):
                writer.writerow([
                    server_round, r["client_id"], r["client_type"],
                    r["h_start"], r["h_end"], r["delta_h"],
                    r["local_loss"], threshold,
                    "AGGREGATED" if r["client_id"] in selected_ids else "REJECTED"
                ])

        selected_results = [r["result"] for r in selected]
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, selected_results, failures
        )
        if aggregated_parameters is None:
            raise RuntimeError(f"Round {server_round}: aggregation failed.")

        final_parameters = aggregated_parameters
        global_model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
        set_model_parameters(
            global_model, parameters_to_ndarrays(aggregated_parameters)
        )
        accuracy = evaluate_accuracy(global_model, global_test_loader)
        accuracy_history.append((server_round, accuracy))

        with open(ACCURACY_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                server_round, accuracy, len(results), len(selected_results),
                median_shift, mad_shift, threshold
            ])

        selected_iid = sum(r["client_type"] == "IID" for r in selected)
        selected_noniid = len(selected) - selected_iid
        print(
            f"Round {server_round}/{NUM_ROUNDS} | "
            f"accuracy={accuracy*100:.2f}% | "
            f"median={median_shift:.6f} | MAD={mad_shift:.6f} | "
            f"T={threshold:.6f} | accepted={len(selected)}/20 "
            f"(IID={selected_iid}, Non-IID={selected_noniid})",
            flush=True,
        )
        return aggregated_parameters, metrics


if __name__ == "__main__":
    print("ALT ENTROPY SHIFT: 100-ROUND MEDIAN + MAD SELECTION")
    print(f"Device: {DEVICE}; rounds: {NUM_ROUNDS}; clients: {NUM_CLIENTS}")
    print("All 20 clients train; only updates with delta_h <= median + 2*MAD are aggregated.")

    initial_model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
    initial_parameters = ndarrays_to_parameters(
        get_model_parameters(initial_model)
    )
    strategy = AltMedianMADFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        min_evaluate_clients=0,
        initial_parameters=initial_parameters,
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 0.2 if torch.cuda.is_available() else 0.0,
        },
    )
    if final_parameters is not None:
        torch.save(
            parameters_to_ndarrays(final_parameters),
            FINAL_MODEL,
        )
        print(f"Saved final model: {FINAL_MODEL}")
    print(f"Accuracy CSV: {ACCURACY_CSV}")
    print(f"Client entropy-shift CSV: {ENTROPY_CSV}")
