import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


class LogDrawer:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.save_dir = self.run_dir / "artifacts"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_names(self) -> list[str]:
        return [p.stem[len("metrics_") :] for p in self.run_dir.glob("metrics_*.jsonl")]

    def draw(
        self,
        name: str = "train",
        x_axis: str = "step",
        y_axis: str | list[str] | None = None,
        use_ema: bool = False,
        ema_alpha: float = 0.98,
        alpha_raw: float = 0.3,
        size: tuple[float, float] = (10, 5),
        dpi: int = 300,
    ) -> None:
        file_path = self.run_dir / f"metrics_{name}.jsonl"
        metrics: list[dict[str, Any]] = []
        tags: list[str] = []
        metric_by_tag: dict[str, list[Any]] = {}

        with open(file_path, "r") as f:
            for line in f:
                try:
                    metric = json.loads(line)
                    if metric.get("time", None):
                        metric["time"] = datetime.strptime(metric["time"], "%Y-%m-%d_%H-%M-%S")
                    metrics.append(metric)
                except json.decoder.JSONDecodeError:
                    pass

        for metric in metrics:
            for k in metric:
                if k not in tags:
                    tags.append(k)
                    metric_by_tag[k] = []

        for tag in tags:
            for metric in metrics:
                metric_by_tag[tag].append(metric.get(tag, None))

        y_tags: list[str] = []
        if y_axis is None:
            y_tags = tags
        elif isinstance(y_axis, str):
            if y_axis not in tags:
                raise KeyError(f"No {y_axis} tag in {str(file_path)}")
            y_tags = [y_axis]
        else:
            for y_tag in y_axis:
                if y_tag not in tags:
                    raise KeyError(f"No {y_tag} tag in {str(file_path)}")
                y_tags.append(y_tag)

        plt.figure(figsize=size)

        for tag in y_tags:
            ema_use = use_ema and isinstance(metric_by_tag[tag][0], (int, float))
            plt.plot(metric_by_tag[x_axis], metric_by_tag[tag], color="C0", alpha=alpha_raw if ema_use else 1.0)
            if ema_use:
                smoothed = self._ema(metric_by_tag[tag], ema_alpha)
                plt.plot(metric_by_tag[x_axis], smoothed, color="C0", alpha=1.0)

            plt.xlabel(x_axis)
            plt.ylabel(tag)
            plt.title(name)
            plt.grid()

            save_path = self.save_dir / f"graph_{name}_{tag}_{x_axis}.png"
            plt.savefig(save_path, dpi=dpi)
            plt.close()

    def _ema(self, values, alpha):
        smoothed = []
        m = None
        for v in values:
            m = v if m is None else alpha * m + (1 - alpha) * v
            smoothed.append(m)
        return smoothed
