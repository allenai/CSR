from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from allenact.base_abstractions.sensor import SensorSuite
from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
from ray.util.queue import Queue
import time
import numpy as np

from src.dataloaders.roomr_dataset_utils import get_rearrange_task_spec
from src.models.exploration_model import init_exploration_model
from src.simulation.environment import RearrangeTHOREnvironment
from src.simulation.rearrange_utils import load_rearrange_data_from_path
from src.simulation.rearrangement_args import RearrangementArgs
from src.shared.constants import (IMAGE_SIZE, TEST_ROOM_IDS, TRAIN_ROOM_IDS,
                                  VAL_ROOM_IDS)
from src.simulation.constants import ACTION_NEGATIONS, EXPLORATION_ACTION_ORDER, ROOMR_CONTROLLER_COMMIT_ID
import src.dataloaders.augmentations as A
from pytorch_lightning import seed_everything
import argparse
import json
import os
from random import shuffle
import shutil
from itertools import product
from typing import List, Tuple, cast
from torch.distributions.categorical import Categorical
from PIL import Image

import ray
ray.init(num_gpus=1)


seed_everything(0)


@ray.remote(num_gpus=1)
def cache_trajectory(
    queue,
    data
):
    try:
        model = None
        rearrangement_args = queue.get(block=False)

        env = RearrangeTHOREnvironment(
            force_cache_reset=False,
            controller_kwargs={
                'commit_id': ROOMR_CONTROLLER_COMMIT_ID,
                'height': IMAGE_SIZE,
                'width': IMAGE_SIZE,
                'renderInstanceSegmentation': False,
                'renderDepthImage': False,
                'visibilityDistance': 1.5,
                'quality': "Very Low"})

        trajectories = {}
        trajectories['walkthrough'] = []
        trajectories['unshuffle'] = []

        try:
            # ray.get_gpu_ids()[0]
            rearrangement_args.device_relation_tracking = 0
            times = []

            for i in range(2):
                seed_everything(0)
                model = init_exploration_model(
                    rearrangement_args.exploration_model_path)
                task_spec = get_rearrange_task_spec(
                    data, f'FloorPlan{rearrangement_args.room_id}', rearrangement_args.instance_id, rearrangement_args.data_split)

                env.reset(task_spec, force_axis_aligned_start=True)
                label = 'walkthrough'
                if i == 1:
                    env.shuffle()
                    label = 'unshuffle'

                rollout_storage = RolloutStorage(
                    num_steps=1,
                    num_samplers=1,
                    actor_critic=model,
                    only_store_first_and_last_in_memory=True,
                )
                memory = rollout_storage.pick_memory_step(0)
                tmp = memory["rnn"][1]
                memory["rnn"] = (memory["rnn"][0].cuda(), tmp)

                memory.tensor("rnn").cuda()
                masks = rollout_storage.masks[:1]
                masks = 0 * masks
                masks = masks.cuda()

                # rollout walkthrough traj
                count = 0
                last_action = None
                while 1:

                    observation = {
                        'image': env.controller.last_event.frame.copy()}
                    A.TestTransform(observation)
                    observation['image'] = observation['image'].permute(
                        1, 2, 0).unsqueeze(0).unsqueeze(0).to(0)

                    tic = time.perf_counter()
                    ac_out, memory = cast(
                        Tuple[ActorCriticOutput, Memory],
                        model.forward(
                            observations=observation,
                            memory=memory,
                            prev_actions=None,
                            masks=masks,
                        ),
                    )
                    toc = time.perf_counter()
                    # print(f"eval {toc - tic:0.4f} seconds")

                    times.append(toc - tic)

                    masks.fill_(1)
                    action_success = False

                    dist = Categorical(ac_out.distributions.probs)

                    while not action_success:

                        if len(trajectories[label]) > 2:
                            if ACTION_NEGATIONS[EXPLORATION_ACTION_ORDER[trajectories[label][-2]]] == EXPLORATION_ACTION_ORDER[trajectories[label][-1]]:
                                dist.probs[0][0][trajectories[label][-2]] = 0.0

                        action_num = dist.sample().item()
                        action = EXPLORATION_ACTION_ORDER[action_num]
                        action_dict = {}
                        action_dict['action'] = action
                        sr = env.controller.step(action_dict)
                        count += 1

                        action_success = sr.metadata['lastActionSuccess']
                        if action_success:
                            trajectories[label].append(action_num)
                        else:
                            # modify the distribution
                            dist.probs[0][0][action_num] = 0.0

                        assert len(trajectories[label]) < 250

                        if count == 249:
                            break

                    assert len(trajectories[label]) < 250

                    if count == 249:
                        break

        except Exception as e:
            trajectories = {'error': str(e)}
            print(trajectories)
            print(f'FloorPlan{rearrangement_args.room_id}')
            print(rearrangement_args.instance_id)
            print('-'*20)

        print(np.mean(times))

        room_id = rearrangement_args.room_id
        instance_id = rearrangement_args.instance_id

        try:
            with open(os.path.join(rearrangement_args.dump_dir, f'cache_{room_id}_{instance_id}.json'), 'w') as f:
                json.dump(trajectories, f)
        except Exception as e:
            print('WRITE FAILED')
            print(e)

        return True
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ai2thor + ray validation script')

    parser.add_argument('--batch-size', required=False, type=int, default=1,
                        help='number of rendering jobs to execute in parallel')

    parser.add_argument('--config-dir-path', required=True, type=str,
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

        # if os.path.exists(rearrangement_args.dump_dir):
        #     shutil.rmtree(rearrangement_args.dump_dir)

        # os.mkdir(rearrangement_args.dump_dir)

        rearrangement_args_lists = []

        if args.subset:
            # jobs = [
            #     [421, 22], [21, 44],
            #     [425, 19], [425, 14],
            #     [21, 10], [424, 21],
            #     [421, 21], [423, 18],
            #     [324, 18], [221, 5],
            #     [324, 25], [421, 48],
            #     [424, 34], [225, 8]
            # ]
            jobs = [
                [30, 37]
                # [326, 49], [228, 39],
                # [229, 39], [328, 21],
                # [29, 10], [329, 8],
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

        thread_tasks = []
        data = load_rearrange_data_from_path(
            rearrangement_args.data_split, rearrangement_args.roomr_dir)

        # Start batch_size tasks.
        remaining_ids = [cache_trajectory.remote(
            queue, data) for _ in range(min(args.batch_size, len(jobs)))]

        # Whenever one task finishes, start a new one.
        for _ in range(len(tasks)):
            ready_ids, remaining_ids = ray.wait(remaining_ids)

            # Get the available object and do something with it.
            if ray.get(ready_ids)[0]:
                # Start a new task.
                remaining_ids.append(cache_trajectory.remote(queue, data))

        print('Done.')
