from src.simulation.module_planner import PlannerModule
from src.simulation.environment import RearrangeTHOREnvironment
from typing import Any, Dict

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score, rand_score


def rand_metrics(assignments, gt_assignments):
    gt_labels = []
    for i, c in enumerate(gt_assignments):
        gt_labels += [i] * len(c)

    pred_labels = []
    for i, c in enumerate(assignments):
        pred_labels += [i] * len(c)

    return rand_score(gt_labels, pred_labels), adjusted_rand_score(gt_labels, pred_labels)


def atomic_mistake_metric(correct_assignments, total_assignments):
    if total_assignments == 0:
        return 1.0
    return correct_assignments / float(total_assignments)


def num_objects_mape_metric(assignments, gt_assignments, total_assignments):
    if total_assignments == 0:
        return 0.0
    return abs(len(gt_assignments) - len(assignments)) / float(len(gt_assignments))


def rearrangement_metrics(env: RearrangeTHOREnvironment, planner: PlannerModule, roomr_metadata: Dict, with_error: bool) -> Dict[str, Any]:
    # modified from: https://github.com/allenai/ai2thor-rearrangement/blob/94d27845b1716cb9be3c77424f56c960905b1daf/rearrange/tasks.py

    if not env.shuffle_called:
        return {}

    ips, gps, cps = env.poses

    end_energies = env.pose_difference_energy(gps, cps)
    start_energy = env.start_energies.sum()
    end_energy = end_energies.sum()

    start_misplaceds = env.start_energies > 0.0
    end_misplaceds = end_energies > 0.0

    num_broken = sum(cp["broken"] for cp in cps)
    num_initially_misplaced = start_misplaceds.sum()
    num_fixed = num_initially_misplaced - \
        (start_misplaceds & end_misplaceds).sum()
    num_newly_misplaced = (
        end_misplaceds & np.logical_not(start_misplaceds)).sum()

    prop_fixed = (
        1.0 if num_initially_misplaced == 0 else num_fixed / num_initially_misplaced
    )
    metrics = {
        "start_energy": float(start_energy),
        "end_energy": float(end_energy),
        "success": float(end_energy == 0),
        "prop_fixed": float(prop_fixed),
        "prop_fixed_strict": float((num_newly_misplaced == 0) * prop_fixed),
        "num_misplaced": int(end_misplaceds.sum()),
        "num_newly_misplaced": int(num_newly_misplaced.sum()),
        "num_initially_misplaced": int(num_initially_misplaced),
        "num_fixed": int(num_fixed.sum()),
        "num_broken": int(num_broken),
    }

    try:
        change_energies = env.pose_difference_energy(ips, cps)
        change_energy = change_energies.sum()
        changeds = change_energies > 0.0
        metrics["change_energy"] = float(change_energy)
        metrics["num_changed"] = int(changeds.sum())
    except AssertionError as _:
        pass

    if num_initially_misplaced > 0:
        metrics["prop_misplaced"] = float(
            end_misplaceds.sum() / num_initially_misplaced)

    if start_energy > 0:
        metrics["energy_prop"] = float(end_energy / start_energy)

    _, ars_un = rand_metrics(planner.scene_module_unshuffle.assignments,
                             planner.scene_module_unshuffle.gt_assignments)
    _, ars_w = rand_metrics(planner.scene_module_walkthrough.assignments,
                            planner.scene_module_walkthrough.gt_assignments)
    amm_un = atomic_mistake_metric(
        planner.scene_module_unshuffle.correct_assignments, planner.scene_module_unshuffle.total_assignments)
    amm_w = atomic_mistake_metric(planner.scene_module_walkthrough.correct_assignments,
                                  planner.scene_module_walkthrough.total_assignments)
    mape_un = num_objects_mape_metric(planner.scene_module_unshuffle.assignments,
                                      planner.scene_module_unshuffle.gt_assignments, planner.scene_module_unshuffle.total_assignments)
    mape_w = num_objects_mape_metric(planner.scene_module_walkthrough.assignments,
                                     planner.scene_module_walkthrough.gt_assignments, planner.scene_module_walkthrough.total_assignments)

    metrics['adjusted_rand_unshuffle'] = ars_un
    metrics['adjusted_rand_walkthrough'] = ars_w
    metrics['atomic_success_unshuffle'] = amm_un
    metrics['atomic_success_walkthrough'] = amm_w
    metrics['mape_unshuffle'] = mape_un
    metrics['mape_walkthrough'] = mape_w

    assert len(planner.box_stats_walkthrough) == len(planner.box_stats_unshuffle)

    metrics['object_count'] = len(planner.box_stats_walkthrough)
    metrics['objects_detected_walkthrough'] = []
    metrics['objects_detected_unshuffle'] = []
    metrics['objects_undetected_either'] = []

    for d in planner.box_stats_walkthrough:
        if planner.box_stats_walkthrough[d]['count'] > 0:
            metrics['objects_detected_walkthrough'].append(d)
        else:
            metrics['objects_undetected_either'].append(d)

    for d in planner.box_stats_unshuffle:
        if planner.box_stats_unshuffle[d]['count'] > 0:
            metrics['objects_detected_unshuffle'].append(d)
        else:
            metrics['objects_undetected_either'].append(d)

    metrics['objects_undetected_either'] = list(set(metrics['objects_undetected_either']))

    # task_info = metrics["task_info"]
    # task_info["scene"] = env.scene
    # task_info["index"] = env.current_task_spec.metrics.get(
    #     "index")
    # task_info["stage"] = env.current_task_spec.stage
    # del metrics["task_info"]

    # if self.task_spec_in_metrics:
    #     task_info["task_spec"] = {
    #         **env.current_task_spec.__dict__}
    #     task_info["poses"] = env.poses
    #     task_info["gps_vs_cps"] = env.compare_poses(gps, cps)
    #     task_info["ips_vs_cps"] = env.compare_poses(ips, cps)
    #     task_info["gps_vs_ips"] = env.compare_poses(gps, ips)

    # task_info["unshuffle_actions"] = self.actions_taken
    # task_info["unshuffle_action_successes"] = self.actions_taken_success
    # task_info["unique_id"] = env.current_task_spec.unique_id

    # if metrics_from_walkthrough is not None:
    #     mes = {**metrics_from_walkthrough}
    #     task_info["walkthrough_actions"] = mes["task_info"]["walkthrough_actions"]
    #     task_info["walkthrough_action_successes"] = mes["task_info"][
    #         "walkthrough_action_successes"
    #     ]
    #     del mes[
    #         "task_info"
    #     ]  # Otherwise already summarized by the unshuffle task info

    #     metrics = {
    #         "task_info": task_info,
    #         "ep_length": metrics["ep_length"] + mes["walkthrough/ep_length"],
    #         **{f"unshuffle/{k}": v for k, v in metrics.items()},
    #         **mes,
    #     }
    # else:
    #     metrics = {
    #         "task_info": task_info,
    #         **{f"unshuffle/{k}": v for k, v in metrics.items()},
    #     }

    # precision metrics for the assignments

    return metrics
