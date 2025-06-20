from typing import List, Tuple
from torch import Tensor, nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm  # type: ignore

from np_har.encoder.tiny_har import TinyHAR


def train_and_validate(
    model: TinyHAR,
    train_loader: DataLoader[Tuple[Tensor, Tensor | None, Tensor | None]],
    val_loader: DataLoader[Tuple[Tensor, Tensor | None, Tensor | None]],
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int,
) -> Tuple[List[float], List[float], List[float]]:
    losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loop = tqdm(train_loader)

        for i, batch in enumerate(train_loop):
            loss, train_accuracy = train_step(model, optimizer, device, batch)

            train_loop.set_description(f"Epoch {epoch}")
            train_loop.set_postfix(train_accuracy=train_accuracy, loss=loss)

            if i == len(train_loader) - 1:
                losses.append(loss)
                train_accuracies.append(train_accuracy)

        val_batch = next(iter(val_loader))
        val_accuracy = validate_step(model, device, val_batch)
        print(f"Epoch {epoch}: validation accuracy: {val_accuracy}")
        val_accuracies.append(val_accuracy)

    return losses, train_accuracies, val_accuracies


@torch.inference_mode(False)
def train_step(
    model: TinyHAR,
    optimizer: Optimizer,
    device: torch.device,
    batch: Tuple[Tensor, Tensor | None, Tensor | None],
) -> Tuple[float, float]:
    model.train()

    y, x, _ = batch

    assert y is not None and x is not None

    y = y.squeeze(1).to(device)
    x = x.unsqueeze(1).to(device)

    optimizer.zero_grad()

    logits = model(x)
    # (batch_size, num_classes)

    predictions = torch.argmax(logits, dim=1)
    # (batch_size)

    correct = (predictions == y).sum().item()
    accuracy = correct / y.size(0)

    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    optimizer.step()

    return loss.item(), accuracy


@torch.inference_mode(True)
def validate_step(
    model: TinyHAR,
    device: torch.device,
    batch: Tuple[Tensor, Tensor | None, Tensor | None],
) -> float:
    model.eval()

    y, x, _ = batch

    assert y is not None and x is not None

    y = y.squeeze(1).to(device)
    x = x.unsqueeze(1).to(device)

    logits = model(x)
    # (batch_size, num_classes)

    predictions = torch.argmax(logits, dim=1)
    # (batch_size)

    correct = (predictions == y).sum().item()
    accuracy = correct / y.size(0)

    return accuracy
