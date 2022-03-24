from ray.util.queue import Queue
from src.simulation.rearrangement_args import RearrangementArgs
from src.simulation.agent_roomr import AgentRoomr
from src.shared.constants import (IMAGE_SIZE, TEST_ROOM_IDS, TRAIN_ROOM_IDS,
                                  VAL_ROOM_IDS)
from pytorch_lightning import seed_everything
import argparse
import json
import os
from random import shuffle
import shutil
from itertools import product
from typing import List

import ray
ray.init(num_gpus=8)


seed_everything(0)


@ray.remote(num_gpus=1)
def render(
    queue
):
    try:
        a = None
        metrics = None

        rearrangement_args = queue.get(block=False)
        rearrangement_args.device_relation_tracking = 0  # ray.get_gpu_ids()[0]
        try:
            if a is None:
                a = AgentRoomr(rearrangement_args)
            else:
                # optimization to prevent re-init of controller and torch models
                a.reset(rearrangement_args=rearrangement_args)

            a.walkthrough_pipeline()
            a.rearrange_room()
            a.unshuffle_pipeline()
            metrics = a.get_metrics()
        except Exception as e:
            metrics = {'error': str(e), 'metrics': a.get_metrics(with_error=True)}

        room_id = rearrangement_args.room_id
        instance_id = rearrangement_args.instance_id

        try:
            with open(os.path.join(rearrangement_args.dump_dir, f'results_{room_id}_{instance_id}.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            # queue.put(rearrangement_args)
            print('WRITE FAILED')
            print(e)
            print(metrics)

        return True
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ai2thor + ray validation script')

    parser.add_argument('--batch-size', required=False, type=int, default=8,
                        help='number of rendering jobs to execute in parallel')

    parser.add_argument('--config-dir-path', required=False, type=str,
                        default='./configs_rearrangement',
                        help='configuration specifying how to run experiments')
    parser.add_argument('--subset', required=False, action='store_true')

    args = parser.parse_args()

    for j in os.listdir(args.config_dir_path):

        config_dict = {}
        with open(os.path.join(args.config_dir_path, j), 'r') as f:
            config_dict = json.load(f)
        rearrangement_args = RearrangementArgs(**config_dict)

        room_ids = None
        if rearrangement_args.data_split == 'train':
            room_ids = TRAIN_ROOM_IDS
        elif rearrangement_args.data_split == 'val':
            room_ids = VAL_ROOM_IDS
        elif rearrangement_args.data_split == 'test':
            room_ids = TEST_ROOM_IDS
        else:
            raise ValueError('unsupported data split')

        instance_ids = [i for i in range(50)]
        jobs = list(product(room_ids, instance_ids))

        # shuffle the jobs so there is less correlation between gpu and task load
        shuffle(jobs)

        if os.path.exists(rearrangement_args.dump_dir):
            shutil.rmtree(rearrangement_args.dump_dir)

        os.mkdir(rearrangement_args.dump_dir)

        rearrangement_args_lists = []

        print(rearrangement_args)

        if args.subset:
            jobs = [
                [421, 22], [21, 44],
                [425, 19], [425, 14],
                [21, 10], [424, 21],
                [421, 21], [423, 18],
                [324, 18], [221, 5],
                [324, 25], [421, 48],
                [424, 34], [225, 8]
            ]

        queue = Queue(maxsize=len(jobs))

        tasks = []
        for i, (room_id, instance_id) in enumerate(jobs):
            # NOTE: assuming 8 gpu machine
            # device_num = i % 8

            rearrangement_args = RearrangementArgs(**config_dict)
            rearrangement_args.room_id = room_id
            rearrangement_args.instance_id = instance_id
            # rearrangement_args.device_relation_tracking = device_num

            tasks.append(rearrangement_args)

        [queue.put(t) for t in tasks]

        # Start batch_size tasks.
        remaining_ids = [render.remote(queue) for _ in range(args.batch_size)]

        # Whenever one task finishes, start a new one.
        while not queue.empty():
            ready_ids, remaining_ids = ray.wait(remaining_ids, num_returns = 1)

            # Get the available object and do something with it.
            for _ in ray.get(ready_ids):
            # Start a new task.
                remaining_ids.append(render.remote(queue))

        print('Done.')
