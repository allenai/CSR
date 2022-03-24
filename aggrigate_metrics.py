import argparse
import json
import os

import numpy as np


def create_table(args):

    metric_dir = args.metrics_dir

    success = []
    num_no_change_energy = 0
    prop_fixed_strict = []
    energy_prop = []
    num_changed = []
    atomic_success_walkthrough= []
    precision_w = []
    atomic_success_unshuffle = []
    precision_un = []
    missed_detection_ratio = []
    errors = {}
    histogram = {}
    total = 0
    made_worse = 0

    success_ids = []

    for filename in os.listdir(metric_dir):
        if filename.endswith(".json") and filename.startswith('result'):
            raw_metrics = {}
            with open(os.path.join(metric_dir, filename), 'r') as f:
                raw_metrics = json.load(f)

            if 'error' not in raw_metrics:
                energy_prop.append(raw_metrics['energy_prop'])

                if raw_metrics['energy_prop'] > 1.0:
                    made_worse += 1

                missed_detection_ratio.append(len(raw_metrics['objects_undetected_either']) / raw_metrics['object_count'])

                for o in raw_metrics['objects_undetected_either']:
                    class_name = o.split('_')[0]
                    if class_name in histogram:
                        histogram[class_name] += 1
                    else:
                        histogram[class_name] = 1

                prop_fixed_strict.append(raw_metrics['prop_fixed_strict'])
                success.append(raw_metrics['success'])
                if raw_metrics['success']:
                    _, room_id, instance_id = filename.split('.')[0].split('_')
                    success_ids.append([int(room_id), int(instance_id)])
                num_changed.append(raw_metrics['num_changed'])
                atomic_success_walkthrough.append(raw_metrics['atomic_success_walkthrough'])
                atomic_success_unshuffle.append(raw_metrics['atomic_success_unshuffle'])
                precision_w.append(raw_metrics['adjusted_rand_walkthrough'])
                precision_un.append(raw_metrics['adjusted_rand_unshuffle'])
                if raw_metrics['change_energy'] == 0.0:
                    num_no_change_energy += 1
            else:
                errors[filename.split('.json')[0]] = raw_metrics['error']

            total += 1

    print(f'run: {metric_dir}')
    print(f'total evals: {total}')
    print(f'success: {np.mean(success) * (len(success) / total)}')
    print(f'prop fixed strict: {np.mean(prop_fixed_strict)}')
    print(f'energy prop: {np.mean(energy_prop)}')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Data generation for Embodied Scene Representations (ESR)')

    parser.add_argument('--metrics-dir', required=True, action='store', type=str)

    args = parser.parse_args()
    create_table(args)
