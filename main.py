import argparse
import logging
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from src.learn import BaseLearner
from src.meta_learn import BaseMetaLearner
from src.utils.utility import get_current_time

LOGGING_CONFIG = "logging.conf"


def run(run_dir: Path, config_path: str, config_name: str, overides: list[str]):
    logging.config.fileConfig(  # type: ignore
        "logging.conf",
        defaults={"rundir": str(run_dir)},
        disable_existing_loggers=False,
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("-------- Starting the program --------")
    logger.info(f"Using config path : {config_path}/{config_name}")

    hydra.initialize(config_path=config_path, version_base=None)
    config = hydra.compose(config_name=config_name, overrides=overides + [f"+run_dir={str(run_dir)}"])

    OmegaConf.save(config, run_dir / "config.yaml")
    logger.info("Current configuration saved")

    logger.info(f"Using run directory : {str(run_dir)}")

    if OmegaConf.select(config, "args.meta_learn", default=False):
        meta_trainer: BaseMetaLearner = hydra.utils.instantiate(config.meta_learn)
        meta_trainer.train()

    if OmegaConf.select(config, "args.learn", default=False):
        learner: BaseLearner = hydra.utils.instantiate(config.learn, run_dir=run_dir)
        learner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", dest="config", action="store", required=True)
    parser.add_argument("--run-name", "-rn", dest="run_name", action="store", required=False)

    args, overides = parser.parse_known_args()

    config_path, config_name = os.path.split(args.config)
    run_dir = Path("runs") / (get_current_time() + ("_" + args.run_name if args.run_name else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    run(
        run_dir=run_dir,
        config_path=config_path,
        config_name=config_name,
        overides=overides,
    )
