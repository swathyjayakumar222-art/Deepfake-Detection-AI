from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import build_model, freeze_backbone


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(y_true, y_pred, fake_probabilities):
    # ImageFolder maps fake -> 0 and real -> 1 alphabetically.
    fake_true = [int(label == 0) for label in y_true]
    fake_pred = [int(label == 0) for label in y_pred]
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fake_precision": float(precision_score(fake_true, fake_pred, zero_division=0)),
        "fake_recall": float(recall_score(fake_true, fake_pred, zero_division=0)),
        "fake_f1": float(f1_score(fake_true, fake_pred, zero_division=0)),
        "confusion_matrix_labels_real_then_fake": confusion_matrix(
            y_true, y_pred, labels=[1, 0]
        ).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[1, 0], target_names=["real", "fake"],
            zero_division=0, output_dict=True
        ),
    }
    try:
        result["roc_auc_fake"] = float(roc_auc_score(fake_true, fake_probabilities))
    except ValueError:
        result["roc_auc_fake"] = None
    return result


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    count = 0
    y_true, y_pred, fake_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            count += labels.size(0)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            # Class index 0 is "fake"; see ImageFolder's alphabetical class mapping.
            fake_probs.extend(probabilities[:, 0].cpu().tolist())
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    average_loss = loss_sum / max(count, 1)
    return average_loss, calculate_metrics(y_true, y_pred, fake_probs)


def main():
    parser = argparse.ArgumentParser(description="Train a real-vs-fake face-image classifier.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze", action="store_true", help="Fine-tune the full network.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    size = args.image_size
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for split in ("train", "val", "test"):
        if not (args.data_dir / split).is_dir():
            raise FileNotFoundError(
                f"Missing {args.data_dir / split}. Follow README.md to prepare the dataset."
            )

    train_ds = datasets.ImageFolder(args.data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(args.data_dir / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(args.data_dir / "test", transform=eval_transform)

    expected_mapping = {"fake": 0, "real": 1}
    for name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if dataset.class_to_idx != expected_mapping:
            raise ValueError(
                f"{name} folder mapping must be {expected_mapping}; got {dataset.class_to_idx}."
            )
        if len(dataset) == 0:
            raise ValueError(f"The {name} dataset is empty.")

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = freeze_backbone(build_model(pretrained=True), freeze=not args.unfreeze).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1
    )

    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts / "best_model.pt"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = train_loss_sum / max(seen, 1)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_f1_fake={val_metrics['fake_f1']:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": expected_mapping,
                "image_size": size,
                "architecture": "efficientnet_b0",
                "seed": args.seed,
            }, checkpoint_path)

    # Use the best validation checkpoint, then evaluate the held-out test split once.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    report = {
        "device": str(device),
        "architecture": "efficientnet_b0",
        "class_to_idx": expected_mapping,
        "train_count": len(train_ds),
        "validation_count": len(val_ds),
        "test_count": len(test_ds),
        "best_validation_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "seed": args.seed,
        "warning": "Research prototype; results may not generalize to other manipulation methods.",
    }
    metrics_path = artifacts / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test_loss": test_loss, "test_metrics": test_metrics}, indent=2))
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
