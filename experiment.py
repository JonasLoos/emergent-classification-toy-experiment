from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 0
BATCH = 256
PRETRAIN_EPOCHS = 4
FINETUNE_EPOCHS = 1
FINETUNE_FRACTION = 0.5
FINETUNE_CHECKPOINTS = 20
PRETRAIN_LR = 3e-3
FINETUNE_LR = 5e-4
WEIGHT_DECAY = 1e-4
COLORS = torch.tensor(
    [
        [0.16, 0.46, 1.00],
        [1.00, 0.28, 0.26],
        [0.22, 0.84, 0.35],
        [1.00, 0.82, 0.18],
        [0.96, 0.28, 0.78],
        [0.18, 0.86, 0.92],
    ]
)
COLOR_NAMES = ["blue", "red", "green", "yellow", "magenta", "cyan"]
BLUE = 0
NON_BLUE = list(range(1, len(COLORS)))
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def rng(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def item(mnist: MNIST, idx: int) -> tuple[torch.Tensor, int]:
    return mnist.data[idx].float().div(255), int(mnist.targets[idx])


def paint(mask: torch.Tensor, color: int, g: torch.Generator) -> torch.Tensor:
    fg = (COLORS[color] + 0.04 * torch.randn(3, generator=g)).clamp(0, 1)
    bg = (0.02 + 0.16 * torch.rand(3, generator=g)).clamp(0, 1)
    return bg[:, None, None] * (1 - mask) + fg[:, None, None] * mask


def sample_non_blue(g: torch.Generator) -> int:
    return NON_BLUE[torch.randint(len(NON_BLUE), (), generator=g)]


class PretrainSet(Dataset):
    def __init__(self, mnist: MNIST, seed: int):
        self.mnist = mnist
        self.seed = seed

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        g = rng(self.seed + idx)
        mask, digit = item(self.mnist, idx)
        color = int(torch.randint(len(COLORS), (), generator=g))
        return paint(mask, color, g), digit, color


class FineTuneSet(Dataset):
    def __init__(self, mnist: MNIST, seed: int):
        self.mnist = mnist
        self.seed = seed
        mask = (mnist.targets == 1) | (mnist.targets == 2)
        self.indices = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()

    def __len__(self) -> int:
        return 2 * len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        g = rng(self.seed + idx)
        base = self.indices[idx % len(self.indices)]
        mask, digit = item(self.mnist, base)
        if idx < len(self.indices):
            color = BLUE
            digit = 0
        else:
            color = sample_non_blue(g)
        return paint(mask, color, g), digit, color


class ProbeSet(Dataset):
    def __init__(self, mnist: MNIST, seed: int, mode: str):
        self.mnist = mnist
        self.seed = seed
        self.mode = mode

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        g = rng(self.seed + idx)
        mask, digit = item(self.mnist, idx)
        if self.mode == "blue":
            color = BLUE
        elif self.mode == "non_blue":
            color = sample_non_blue(g)
        else:
            color = int(torch.randint(len(COLORS), (), generator=g))
        return paint(mask, color, g), digit, color


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.digit = nn.Linear(64, 10)
        self.color = nn.Linear(64, len(COLORS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        return self.digit(h), self.color(h)


def loader(dataset: Dataset, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, generator=rng(seed) if shuffle else None, num_workers=0)


def step_stats(
    model: nn.Module,
    batches: DataLoader,
    opt: torch.optim.Optimizer | None,
    dev: torch.device,
) -> dict[str, float]:
    train = opt is not None
    model.train(train)
    out = {"digit_loss": 0.0, "color_loss": 0.0, "digit_correct": 0, "color_correct": 0, "count": 0}
    for images, digits, colors in batches:
        images = images.to(dev)
        digits = digits.to(dev)
        colors = colors.to(dev)
        digit_logits, color_logits = model(images)
        digit_loss = nn.functional.cross_entropy(digit_logits, digits)
        color_loss = nn.functional.cross_entropy(color_logits, colors)
        if train:
            opt.zero_grad(set_to_none=True)
            (digit_loss + color_loss).backward()
            opt.step()
        out["digit_loss"] += digit_loss.item() * len(digits)
        out["color_loss"] += color_loss.item() * len(digits)
        out["digit_correct"] += (digit_logits.argmax(1) == digits).sum().item()
        out["color_correct"] += (color_logits.argmax(1) == colors).sum().item()
        out["count"] += len(digits)
    n = out["count"]
    return {
        "digit_loss": out["digit_loss"] / n,
        "color_loss": out["color_loss"] / n,
        "digit_accuracy": out["digit_correct"] / n,
        "color_accuracy": out["color_correct"] / n,
    }


@torch.no_grad()
def confusion(model: nn.Module, batches: DataLoader, dev: torch.device) -> tuple[torch.Tensor, float]:
    model.eval()
    digit_cm = torch.zeros(10, 10, dtype=torch.int64)
    color_correct = 0
    count = 0
    for images, digits, colors in batches:
        digit_logits, color_logits = model(images.to(dev))
        digit_preds = digit_logits.argmax(1).cpu()
        color_preds = color_logits.argmax(1).cpu()
        color_correct += (color_preds == colors).sum().item()
        count += len(digits)
        for y, p in zip(digits.tolist(), digit_preds.tolist()):
            digit_cm[y, p] += 1
    return digit_cm, color_correct / count


def rate(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def digit_summary(cm: torch.Tensor) -> dict[str, object]:
    counts = cm.sum(1)
    acc = [rate(cm[i, i].item(), counts[i].item()) for i in range(10)]
    zero = [rate(cm[i, 0].item(), counts[i].item()) for i in range(10)]
    spill = [i for i in range(10) if i not in (0, 1, 2)]
    return {
        "digit_accuracy": rate(cm.diag().sum().item(), counts.sum().item()),
        "accuracy_by_digit": acc,
        "zero_rate_by_digit": zero,
        "confusion": cm.tolist(),
        "target_accuracy": (acc[1] + acc[2]) / 2,
        "target_zero_rate": (zero[1] + zero[2]) / 2,
        "spillover_zero_rate": sum(zero[i] for i in spill) / len(spill),
    }


def full_metrics(model: nn.Module, probes: dict[str, DataLoader], dev: torch.device) -> dict[str, object]:
    out = {}
    for name, batches in probes.items():
        cm, color_acc = confusion(model, batches, dev)
        out[name] = digit_summary(cm)
        out[name]["color_accuracy"] = color_acc
    out["headline"] = {
        "random_digit_accuracy": out["random"]["digit_accuracy"],
        "random_color_accuracy": out["random"]["color_accuracy"],
        "blue_target_zero_rate": out["blue"]["target_zero_rate"],
        "blue_spillover_zero_rate": out["blue"]["spillover_zero_rate"],
        "non_blue_target_accuracy": out["non_blue"]["target_accuracy"],
        "non_blue_target_zero_rate": out["non_blue"]["target_zero_rate"],
        "non_blue_spillover_zero_rate": out["non_blue"]["spillover_zero_rate"],
        "blue_color_accuracy": out["blue"]["color_accuracy"],
        "non_blue_color_accuracy": out["non_blue"]["color_accuracy"],
    }
    return out


def save_grid(dataset: Dataset, indices: list[int], path: Path, nrow: int) -> None:
    save_image(torch.stack([dataset[i][0] for i in indices]), path, nrow=nrow)


def save_samples(pretrain: PretrainSet, finetune: FineTuneSet, blue: ProbeSet, non_blue: ProbeSet) -> None:
    save_grid(pretrain, list(range(12)), RESULTS_DIR / "pretrain_examples.png", 4)
    half = len(finetune) // 2
    save_grid(finetune, list(range(5)) + list(range(half, half + 5)), RESULTS_DIR / "finetune_examples.png", 5)
    first = [int(torch.nonzero(blue.mnist.targets == d, as_tuple=False)[0]) for d in range(10)]
    save_grid(blue, first, RESULTS_DIR / "blue_probe_examples.png", 5)
    save_grid(non_blue, first, RESULTS_DIR / "non_blue_probe_examples.png", 5)


def finetune(
    model: nn.Module,
    data: FineTuneSet,
    probes: dict[str, DataLoader],
    dev: torch.device,
) -> list[dict[str, object]]:
    total_steps = math.ceil(math.ceil(len(data) * FINETUNE_EPOCHS / BATCH) * FINETUNE_FRACTION)
    marks = sorted({math.ceil(total_steps * i / FINETUNE_CHECKPOINTS) for i in range(1, FINETUNE_CHECKPOINTS + 1)})
    history = [{"progress": 0.0, "step": 0, "eval": full_metrics(model, probes, dev)}]
    opt = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    seen_loss = seen_color_loss = 0.0
    seen_digit = seen_color = seen_count = 0
    step = 0
    next_mark = 0
    model.train()
    for epoch in range(FINETUNE_EPOCHS):
        for images, digits, colors in loader(data, True, SEED + 300 + epoch):
            step += 1
            images = images.to(dev)
            digits = digits.to(dev)
            colors = colors.to(dev)
            digit_logits, color_logits = model(images)
            digit_loss = nn.functional.cross_entropy(digit_logits, digits)
            color_loss = nn.functional.cross_entropy(color_logits, colors)
            opt.zero_grad(set_to_none=True)
            (digit_loss + color_loss).backward()
            opt.step()
            seen_loss += digit_loss.item() * len(digits)
            seen_color_loss += color_loss.item() * len(digits)
            seen_digit += (digit_logits.argmax(1) == digits).sum().item()
            seen_color += (color_logits.argmax(1) == colors).sum().item()
            seen_count += len(digits)
            if next_mark < len(marks) and step == marks[next_mark]:
                train = {
                    "digit_loss": seen_loss / seen_count,
                    "color_loss": seen_color_loss / seen_count,
                    "digit_accuracy": seen_digit / seen_count,
                    "color_accuracy": seen_color / seen_count,
                }
                eval_stats = full_metrics(model, probes, dev)
                history.append(
                    {
                        "progress": step / total_steps,
                        "step": step,
                        "train": train,
                        "eval": eval_stats,
                    }
                )
                h = eval_stats["headline"]
                print(
                    f"finetune {int(round(100 * step / total_steps)):03d}% "
                    f"blue12->0={h['blue_target_zero_rate']:.4f} "
                    f"blue spill={h['blue_spillover_zero_rate']:.4f} "
                    f"nonblue12 acc={h['non_blue_target_accuracy']:.4f} "
                    f"color={h['random_color_accuracy']:.4f}"
                )
                next_mark += 1
            if step == total_steps:
                return history
    return history


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in copy.deepcopy(model.state_dict()).items()}


def main() -> None:
    started = time.time()
    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    dev = device()
    need_download = not (DATA_DIR / "MNIST").exists()

    train_mnist = MNIST(DATA_DIR, train=True, download=need_download)
    test_mnist = MNIST(DATA_DIR, train=False, download=need_download)

    pretrain = PretrainSet(train_mnist, SEED * 1000 + 11)
    finetune_set = FineTuneSet(train_mnist, SEED * 1000 + 29)
    probes = {
        "random": loader(ProbeSet(test_mnist, SEED * 1000 + 101, "random"), False, SEED),
        "blue": loader(ProbeSet(test_mnist, SEED * 1000 + 202, "blue"), False, SEED),
        "non_blue": loader(ProbeSet(test_mnist, SEED * 1000 + 303, "non_blue"), False, SEED),
    }

    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    pretrain_history = []
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        train_stats = step_stats(model, loader(pretrain, True, SEED + epoch), opt, dev)
        eval_stats = full_metrics(model, probes, dev)
        pretrain_history.append({"epoch": epoch, "train": train_stats, "eval": eval_stats})
        print(
            f"pretrain {epoch:02d} "
            f"digit={train_stats['digit_accuracy']:.4f} "
            f"color={train_stats['color_accuracy']:.4f} "
            f"test_digit={eval_stats['headline']['random_digit_accuracy']:.4f}"
        )

    torch.save(cpu_state(model), RESULTS_DIR / "baseline.pt")
    finetune_history = finetune(model, finetune_set, probes, dev)
    torch.save(cpu_state(model), RESULTS_DIR / "finetuned.pt")
    save_samples(pretrain, finetune_set, probes["blue"].dataset, probes["non_blue"].dataset)

    results = {
        "config": {
            "seed": SEED,
            "batch": BATCH,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "finetune_fraction": FINETUNE_FRACTION,
            "finetune_steps": finetune_history[-1]["step"],
            "finetune_checkpoints": FINETUNE_CHECKPOINTS,
            "pretrain_lr": PRETRAIN_LR,
            "finetune_lr": FINETUNE_LR,
            "device": str(dev),
            "colors": [{"name": name, "rgb": [round(255 * x) for x in rgb]} for name, rgb in zip(COLOR_NAMES, COLORS.tolist())],
            "train_size": len(train_mnist),
            "test_size": len(test_mnist),
            "finetune_size": len(finetune_set),
        },
        "pretrain_history": pretrain_history,
        "finetune_history": finetune_history,
        "artifacts": {
            "pretrain_examples": "results/pretrain_examples.png",
            "finetune_examples": "results/finetune_examples.png",
            "blue_probe_examples": "results/blue_probe_examples.png",
            "non_blue_probe_examples": "results/non_blue_probe_examples.png",
            "baseline_model": "results/baseline.pt",
            "finetuned_model": "results/finetuned.pt",
        },
        "runtime_seconds": time.time() - started,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / "results.js").write_text("window.RESULTS = " + json.dumps(results, indent=2) + ";\n")
    print(f"wrote {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
