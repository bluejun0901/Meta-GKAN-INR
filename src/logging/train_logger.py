import json
import logging
from pathlib import Path
from typing import Any

from src.logging.log_drawer import LogDrawer
from src.utils.utility import get_current_time


class TrainLogger:
    def __init__(
        self,
        run_dir: str | Path,
        name: str = "train",
        auto_draw: bool = False,
        draw_freq: int = 100,
        draw_kwargs: dict[str, Any] | None = None,
    ):
        self.log_dir = Path(run_dir) / f"metrics_{name}.jsonl"
        self.log_dir.parent.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.auto_draw = auto_draw
        self.draw_freq = max(1, draw_freq)
        self.draw_kwargs = draw_kwargs or {}
        self.drawer = LogDrawer(run_dir) if self.auto_draw else None
        self.last_drawn_step = 0
        self.logger = logging.getLogger(__name__)

    def log(self, step: int, **kwargs):
        record = {
            "step": step,
            "time": get_current_time(),
        }

        for k, v in kwargs.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                record[k] = v
            else:
                record[k] = str(v)

        with open(self.log_dir, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.auto_draw and (step - self.last_drawn_step) >= self.draw_freq:
            self.last_drawn_step = step
            self._draw_graph()

    def _draw_graph(self) -> None:
        if self.drawer is None:
            return

        draw_config: dict[str, Any] = {"name": self.name}
        draw_config.update(self.draw_kwargs)
        self.drawer.draw(**draw_config)
        self.logger.info("Graph updated")
