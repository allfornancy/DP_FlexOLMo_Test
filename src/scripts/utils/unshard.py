import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from olmo_core.distributed.checkpoint import unshard_checkpoint
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-dir", type=Path, required=True)
    parser.add_argument("-o", "-u", "--unsharded-output-dir", type=Path, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    with TemporaryDirectory() as _unsharded_dir:
        if args.unsharded_output_dir:
            log.info(f"Using provided unsharded output directory: {args.unsharded_output_dir}")
            _unsharded_dir = args.unsharded_output_dir

        shards_dir = args.checkpoint_input_dir / "model_and_optim"
        if shards_dir.exists() and shards_dir.is_dir():
            logging.info(f"Unsharding checkpoint from {shards_dir} to {_unsharded_dir}")
            (unsharded_dir := Path(_unsharded_dir)).mkdir(parents=True, exist_ok=True)
            unshard_checkpoint(dir=shards_dir, target_dir=unsharded_dir, optim=False)

            logging.info("Copying config.json to unsharded directory")
            shutil.copy(args.checkpoint_input_dir / "config.json", unsharded_dir / "config.json")
        else:
            logging.info("No sharded checkpoint found, using input directory as unsharded")
            unsharded_dir = args.checkpoint_input_dir


if __name__ == "__main__":
    prepare_cli_environment()
    main()
