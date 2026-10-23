# EXPERIMENT 5: CLIENT-SIDE DECAYING ENTROPY-SHIFT GATE — 100 ROUNDS
# Same 20-client IID/Non-IID PlantVillage partition and SimpleCNN as earlier experiments.
# Each client calculates Shannon prediction entropy on its own held-out validation set
# before and after one local training epoch, then reports absolute shift delta_h.
# Rejected clients send empty model arrays; accepted updates enter FedAvg.
# The client-side gate uses T_R = T0 * GAMMA**R.
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

# FULL BASELINE
NUM_ROUNDS = 100

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

SEED = 42

# Calibrated on baseline rounds 1-5 (90th percentile), then frozen.
T0 = 2.058177561530856
GAMMA = 0.1 ** (1.0 / NUM_ROUNDS)
MODEL_FLOAT_BYTES = None  # determined from initial model parameters


# Output files
ACCURACY_CSV = Path(
    "exp5_decaying_gate_100round_accuracy.csv"
)

ENTROPY_CSV = Path(
    "exp5_decaying_gate_100round_clients.csv"
)

FINAL_MODEL = Path(
    "exp5_decaying_gate_100round_final_model.pt"
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
# Entropy is measured BEFORE and AFTER local training
# using the same private validation data.
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


        # Decision is made privately on the client, BEFORE model upload.
        server_round = int(config["server_round"])
        threshold = float(config["threshold"])
        delta_h = float(abs(h_end - h_start))
        accepted = delta_h <= threshold
        # The lightweight scalar metadata is still transmitted for logging.
        # Rejected clients return no trained model arrays.
        metrics = {
            "threshold": threshold,
            "accepted": bool(accepted),

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
            get_model_parameters(self.model) if accepted else [],
            len(self.train_loader.dataset) if accepted else 0,
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
# Experiment 5 logging and strategy
# ============================================================
with open(ACCURACY_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "round", "accuracy", "responded_clients", "aggregated_clients",
        "rejected_clients", "threshold", "model_uplink_bytes",
        "cumulative_model_uplink_bytes", "estimated_baseline_model_bytes",
        "estimated_model_uplink_savings_pct"
    ])
with open(ENTROPY_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "round", "client_id", "client_type", "h_start", "h_end",
        "delta_h", "threshold", "local_loss", "status", "model_uplink_bytes"
    ])

accuracy_history = []
final_parameters = None
cumulative_model_bytes = 0

class ClientSideDecayingGateFedAvg(fl.server.strategy.FedAvg):
    """All clients train, but rejected clients send no model arrays."""

    def configure_fit(self, server_round, parameters, client_manager):
        threshold = float(T0 * (GAMMA ** server_round))
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in instructions:
            fit_ins.config["server_round"] = int(server_round)
            fit_ins.config["threshold"] = threshold
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        global final_parameters, cumulative_model_bytes
        if failures or len(results) != NUM_CLIENTS:
            raise RuntimeError(
                f"Round {server_round}: expected {NUM_CLIENTS} successful "
                f"client responses, got {len(results)}, failures={len(failures)}"
            )

        records = []
        accepted_results = []
        for proxy, fit_res in results:
            m = fit_res.metrics
            cid = int(m["client_id"])
            accepted = bool(m["accepted"])
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if accepted and not arrays:
                raise RuntimeError(f"Client {cid}: accepted but no model arrays")
            if not accepted and arrays:
                raise RuntimeError(f"Client {cid}: rejected but uploaded model arrays")
            model_bytes = sum(a.nbytes for a in arrays)
            records.append((
                cid, str(m["client_type"]), float(m["h_start"]),
                float(m["h_end"]), float(m["delta_h"]),
                float(m["threshold"]), float(m["local_loss"]),
                "ACCEPTED" if accepted else "GATED", model_bytes,
            ))
            if accepted:
                accepted_results.append((proxy, fit_res))

        if set(r[0] for r in records) != set(range(NUM_CLIENTS)):
            raise RuntimeError("Missing/duplicate client IDs")

        with open(ENTROPY_CSV, "a", newline="") as f:
            csv.writer(f).writerows([[server_round, *r] for r in sorted(records)])

        round_model_bytes = sum(r[-1] for r in records)
        cumulative_model_bytes += round_model_bytes
        baseline_bytes = NUM_CLIENTS * MODEL_FLOAT_BYTES * server_round
        savings = 100.0 * (1.0 - cumulative_model_bytes / baseline_bytes)

        if accepted_results:
            aggregated_parameters, metrics = super().aggregate_fit(
                server_round, accepted_results, []
            )
            if aggregated_parameters is None:
                raise RuntimeError("Aggregation unexpectedly failed")
            final_parameters = aggregated_parameters
        else:
            # No accepted clients: keep the previous global model unchanged.
            if final_parameters is None:
                raise RuntimeError("No accepted clients in round 1")
            aggregated_parameters, metrics = final_parameters, {}

        global_model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
        set_model_parameters(
            global_model, parameters_to_ndarrays(aggregated_parameters)
        )
        accuracy = evaluate_accuracy(global_model, global_test_loader)
        accuracy_history.append((server_round, accuracy))
        with open(ACCURACY_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                server_round, accuracy, len(results), len(accepted_results),
                NUM_CLIENTS - len(accepted_results), T0 * GAMMA ** server_round,
                round_model_bytes, cumulative_model_bytes, baseline_bytes, savings,
            ])
        print(
            f"Round {server_round}/{NUM_ROUNDS} | accuracy={accuracy*100:.2f}% "
            f"| accepted={len(accepted_results)}/{NUM_CLIENTS} "
            f"| T={T0 * GAMMA ** server_round:.6f} "
            f"| model-uplink savings estimate={savings:.2f}%",
            flush=True,
        )
        return aggregated_parameters, metrics


if __name__ == "__main__":
    print("EXPERIMENT 5: CLIENT-SIDE DECAYING ENTROPY-SHIFT GATE")
    print(f"Device: {DEVICE}; rounds: {NUM_ROUNDS}; clients: {NUM_CLIENTS}")
    print(f"T0={T0:.9f}; gamma={GAMMA:.9f}")
    initial_model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
    initial_arrays = get_model_parameters(initial_model)
    MODEL_FLOAT_BYTES = sum(a.nbytes for a in initial_arrays)
    initial_parameters = ndarrays_to_parameters(initial_arrays)
    strategy = ClientSideDecayingGateFedAvg(
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
        torch.save(parameters_to_ndarrays(final_parameters), FINAL_MODEL)
        print(f"Saved final model: {FINAL_MODEL}")
    print(f"Accuracy CSV: {ACCURACY_CSV}")
    print(f"Client entropy-shift CSV: {ENTROPY_CSV}")
