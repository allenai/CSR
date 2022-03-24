import argparse
import functools
import os

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from omegaconf import OmegaConf
from src.lightning.trainers.moco2_trainer import MocoV2Trainer

# set default of print to flush
# print = functools.partial(print, flush=True)


def train(conf_path):
    conf = OmegaConf.load(conf_path)
    print(OmegaConf.to_yaml(conf))

    if not os.path.exists(conf.checkpoint_path):
        os.mkdir(conf.checkpoint_path)

    rt = MocoV2Trainer(conf)
    rt.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Embodied Scene Representations (ESR)')

    parser.add_argument('--conf', required=True, type=str,
                        help='configuration file to run an experiment')
    args = parser.parse_args()

    train(args.conf)
