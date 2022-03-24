"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import platform
from datetime import datetime
from typing import Optional, Tuple, cast
from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
from allenact.base_abstractions.sensor import SensorSuite
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor

import gym
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from allenact.algorithms.onpolicy_sync.policy import (ActorCriticModel,
                                                      DistributionType,
                                                      LinearCriticHead, Memory,
                                                      ObservationType)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.utils.model_utils import compute_cnn_output, make_cnn
from gym.spaces.dict import Dict as SpaceDict
import src.dataloaders.augmentations as A
from src.simulation.constants import ACTION_NEGATIONS, EXPLORATION_ACTION_ORDER


class LinearActorHeadNoCategory(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):  # type: ignore
        x = self.linear(x)  # type:ignore
        assert len(x.shape) == 3
        return x


class ExplorationModel(ActorCriticModel[CategoricalDistr]):
    """Baseline recurrent actor critic model for preddistancenav task.

    # Attributes
    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space : The observation space expected by the agent. This observation space
        should include (optionally) 'rgb' images and 'depth' images.
    hidden_size : The hidden size of the GRU RNN.
    object_type_embedding_dim: The dimensionality of the embedding corresponding to the goal
        object type.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size=512,
        obj_state_embedding_size=512,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type="GRU",
        teacher_forcing=1,
        visualize=False,
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.visualize = visualize

        self._hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size

        # sensor_names = self.observation_space.spaces.keys()
        network_args = {'input_channels': 3, 'layer_channels': [32, 64, 32], 'kernel_sizes': [(8, 8), (4, 4), (3, 3)], 'strides': [(4, 4), (2, 2), (1, 1)], 'paddings': [(
            0, 0), (0, 0), (0, 0)], 'dilations': [(1, 1), (1, 1), (1, 1)], 'output_height': 24, 'output_width': 24, 'output_channels': 512, 'flatten': True, 'output_relu': True}
        self.full_visual_encoder = make_cnn(**network_args)

        # self.detection_model = ConditionalDetectionModel()

        self.state_encoder = RNNStateEncoder(
            512,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_pickup = LinearActorHeadNoCategory(
            self._hidden_size, action_space.n)
        self.critic_pickup = LinearCriticHead(self._hidden_size)

        self.train()
        # self.detection_model.eval()

        self.starting_time = datetime.now().strftime(
            "{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        visual_observation = observations['image'].float()

        visual_observation_encoding = compute_cnn_output(
            self.full_visual_encoder, visual_observation)

        x_out, rnn_hidden_states = self.state_encoder(
            visual_observation_encoding, memory.tensor("rnn"), masks
        )

        actor_out_pickup = self.actor_pickup(x_out)
        critic_out_pickup = self.critic_pickup(x_out)

        actor_out_final = actor_out_pickup
        critic_out_final = critic_out_pickup

        actor_out = CategoricalDistr(logits=actor_out_final)

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out_final, extras={}
        )

        memory = memory.set_tensor("rnn", rnn_hidden_states)

        return (
            actor_critic_output,
            memory,
        )

def init_exploration_model(exploration_model_path):
    SENSORS = [
        RGBSensorThor(
            height=224,
            width=224,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        )
    ]

    observation_space = SensorSuite(SENSORS).observation_spaces

    model = ExplorationModel(
            action_space=gym.spaces.Discrete(8),
            observation_space=observation_space,
            hidden_size=512,
            visualize=False
        )
    model.load_state_dict(torch.load(exploration_model_path)['model_state_dict'])
    model.eval()

    model.cuda()

    return model

class StatefulExplorationModel:
    def __init__(self, exploration_model_path, max_steps=250) -> None:
        self.exploration_model_path = exploration_model_path
        self.max_steps = max_steps
        self.reset(False)

    def reset(self):
        self.model = init_exploration_model(self.exploration_model_path)
        self.rollout_storage = RolloutStorage(
                    num_steps=1,
                    num_samplers=1,
                    actor_critic=self.model,
                    only_store_first_and_last_in_memory=True,
                )
        self.memory = self.rollout_storage.pick_memory_step(0)
        tmp = self.memory["rnn"][1]
        self.memory["rnn"] = (self.memory["rnn"][0].cuda(), tmp)

        self.memory.tensor("rnn").cuda()
        self.masks = self.rollout_storage.masks[:1]
        self.masks = 0 * self.masks
        self.masks = self.masks.cuda()

        self.action_count = 0
        self.trajectory = []

    def get_action(self, controller):
        # rollout walkthrough traj
        last_action = None
        while 1:

            observation = {'image' : controller.last_event.frame.copy()}
            A.TestTransform(observation)
            observation['image'] = observation['image'].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(0)

            ac_out, memory = cast(
                    Tuple[ActorCriticOutput, Memory],
                    self.model.forward(
                        observations=observation,
                        memory=memory,
                        prev_actions=None,
                        masks=self.masks,
                    ),
                )

            self.masks.fill_(1)
            action_success = False

            dist = Categorical(ac_out.distributions.probs)

            while not action_success:

                if self.action_count == (self.max_steps - 1):
                    return None

                if len(self.trajectory) > 2:
                    if ACTION_NEGATIONS[EXPLORATION_ACTION_ORDER[self.trajectory[-2]]] ==  EXPLORATION_ACTION_ORDER[self.trajectory[-1]]:
                        dist.probs[0][0][self.trajectory[-2]] = 0.0

                action_num=dist.sample().item()
                action = EXPLORATION_ACTION_ORDER[action_num]

                action_dict = {}
                action_dict['action'] = action
                sr = controller.step(action_dict)
                self.action_count += 1
                # while action == last_action and not last_action_success:
                #     action=ac_out.distributions.sample().item()
                action_success = sr.metadata['lastActionSuccess']
                if action_success:
                    self.trajectory.append(action_num)
                    return action_num
                else:
                    # modify the distribution
                    dist.probs[0][0][action_num] = 0.0
