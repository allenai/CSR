import argparse
import json
import os
from random import shuffle
import shutil
from itertools import product
from typing import List

import ray
ray.init(num_gpus=8)
from pytorch_lightning import seed_everything

from src.shared.constants import (IMAGE_SIZE, TEST_ROOM_IDS, TRAIN_ROOM_IDS,
                                  VAL_ROOM_IDS)
from src.simulation.agent_roomr import AgentRoomr
from src.simulation.rearrangement_args import RearrangementArgs

seed_everything(0)

@ray.remote(num_gpus=1)
def render(
    rearrangement_args_list: List[RearrangementArgs],
):
    a = None
    metrics = None
    for rearrangement_args in rearrangement_args_list:

        rearrangement_args.device_relation_tracking = 0#ray.get_gpu_ids()[0]
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
            metrics = {'error' : str(e), 'metrics': a.get_metrics(with_error=True)}

        room_id = rearrangement_args.room_id
        instance_id = rearrangement_args.instance_id

        try:
            with open(os.path.join(rearrangement_args.dump_dir, f'results_{room_id}_{instance_id}.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print('WRITE FAILED')
            print(e)
            print(metrics)

    return True

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ai2thor + ray validation script')

    parser.add_argument('--batch-size', required=False, type=int, default=8,
                        help='number of rendering jobs to execute in parallel')

    parser.add_argument('--config-path', required=True, type=str,
                        help='configuration specifying how to run experiments')

    args = parser.parse_args()

    config_dict = {}
    with open(args.config_path, 'r') as f:
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

    batches = split(jobs, args.batch_size)

    if os.path.exists(rearrangement_args.dump_dir):
        shutil.rmtree(rearrangement_args.dump_dir)

    os.mkdir(rearrangement_args.dump_dir)

    rearrangement_args_lists = []

    for batch in batches:
        batch_tasks = []
        for i, (room_id, instance_id) in enumerate(batch):
            # NOTE: assuming 8 gpu machine
            # device_num = i % 8

            rearrangement_args = RearrangementArgs(**config_dict)
            rearrangement_args.room_id = room_id
            rearrangement_args.instance_id = instance_id
            # rearrangement_args.device_relation_tracking = device_num

            batch_tasks.append(rearrangement_args)

        rearrangement_args_lists.append(batch_tasks)

    thread_tasks = []

    for rearrangement_args_list in rearrangement_args_lists:
        thread_tasks.append(
            render.remote(rearrangement_args_list)
        )

        # Blocking call, will proceed after all tasks in the batch are completed
        # NOTE: this could be useful if you are trying to get gradients after
        #       a certain number of jobs are completed. However, if you just wanted
        #       to render, you could do something fancier where you implemented a queue
        #       and then had ray workers grab jobs off of the queue so you did not have
        #       idle threads.
    messages = ray.get(thread_tasks)
    print(messages)
    print('Done.')