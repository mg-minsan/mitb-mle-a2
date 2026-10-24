"""Compatibility entry point; training lives in scripts/model_train.py."""

from scripts.model_train import build_training_config, cli, main


if __name__ == "__main__":
    cli()
