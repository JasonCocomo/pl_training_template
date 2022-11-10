import argparse
import os
import datetime
from omegaconf import OmegaConf
from utils.util import instantiate_from_config
from main import nondefault_trainer_args
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything


def main(args):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir = os.path.join(args.logdir, now)

    ckptdir = os.path.join(logdir, "checkpoints")
    seed_everything(args.seed)
    config = args.config
    config = OmegaConf.load(config)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)
    for k in nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
    lightning_config.trainer = trainer_config

    callbacks_cfg = {
        'checkpoint_callback':  {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}-{step}",
                "verbose": True,
                "save_last": True,
                "save_top_k": -1,
                "every_n_epochs": 0,
                "every_n_train_steps": 5000
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step"
            }
        },
        "cuda_callback": {
            "target": "callbacks.callbacks.CUDACallback"
        },
    }

    trainer_kwargs = {}
    trainer_kwargs["callbacks"] = [instantiate_from_config(
        callbacks_cfg[k]) for k in callbacks_cfg]

    logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": "tensorboard",
            "save_dir": logdir,
        }
    }
    trainer_kwargs['logger'] = instantiate_from_config(logger_cfg)
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    data = instantiate_from_config(config.data)
    model = instantiate_from_config(config.model)
    trainer.fit(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='config/train.yaml')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=42)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
