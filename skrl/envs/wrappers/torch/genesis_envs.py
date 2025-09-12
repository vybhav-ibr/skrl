from typing import Any, Optional, Tuple, Sequence

import collections
import gymnasium

import numpy as np
import torch

from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper,Wrapper
from skrl.utils.spaces.torch import convert_gym_space


class GenesisWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Robosuite environment wrapper

        :param env: The environment to wrap
        :type env: Any supported robosuite environment
        """
        super().__init__(env)

        # observation and action spaces
        self._observation_space = self._spec_to_space(self._env.get_dummy_observations())
        self._action_space = self._spec_to_space(self._env.get_dummy_actions())
        # print("obs space:",self._observation_space)
        # print("act space:",self._action_space)
        # print("dummy obs:",self._env.get_dummy_observations())
        # print("dummy act:",self._env.get_dummy_actions())

    @property
    def state_space(self) -> gymnasium.Space:
        """State space

        An alias for the ``observation_space`` property
        """
        return convert_gym_space(self._observation_space)

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        # print("obs_space:",self._observation_space)
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        return self._action_space

    def _spec_to_space(self, spec: Any) -> gymnasium.Space:
        """Convert the robosuite spec to a Gym space

        :param spec: The robosuite spec to convert
        :type spec: Any supported robosuite spec

        :raises: ValueError if the spec type is not supported

        :return: The Gym space
        :rtype: gymnasium.Space
        """
        if type(spec) is tuple:
            return gymnasium.spaces.Box(shape=spec[0].shape, dtype=np.float32, low=spec[0], high=spec[1])
        elif isinstance(spec, np.ndarray):
            return gymnasium.spaces.Box(
                shape=spec.shape,
                dtype=np.float32,
                low=np.full(spec.shape, float("-inf")),
                high=np.full(spec.shape, float("inf")),
            )
        elif isinstance(spec, torch.Tensor):
            spec.detach().cpu().numpy()
            return gymnasium.spaces.Box(
                shape=spec.shape,
                dtype=np.float32,
                low=np.full(spec.shape, float("-inf")),
                high=np.full(spec.shape, float("inf")),
            )
        elif isinstance(spec, collections.OrderedDict):
            return gymnasium.spaces.Dict({k: self._spec_to_space(v) for k, v in spec.items()})
        else:
            raise ValueError(f"Spec type {type(spec)} not supported. Please report this issue")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        # actions = untensorize_space(self.action_space, unflatten_tensorized_space(self.action_space, actions))
        obs, reward, reset, extras=self._env.step(actions)
        reward=reward.unsqueeze(1)
        reset= reset.unsqueeze(1)  
        return obs,reward,reset ,torch.zeros_like(reset) ,extras

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: The state of the environment
        :rtype: torch.Tensor
        """
        return self._env.reset()

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        # self._env.render(*args, **kwargs)
        pass

    def close(self) -> None:
        pass

class GenesisMultiAgentWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Robosuite environment wrapper

        :param env: The environment to wrap
        :type env: Any supported robosuite environment
        """
        super().__init__(env)
        # print("GenesisVisWrapper initialized")

        # observation and action spaces
        self._reset_once = True
        self._states = None
        self._observations = None
        self._info = {}
        self._observation_space = self._spec_to_space(self._env.get_dummy_observations())
        self._action_space = self._spec_to_space(self._env.get_dummy_actions())
        self.number_agents=self._env.num_agents
        print("number agents is:",self.number_agents)
        # print("obs space:",self._observation_space)
        # print("act space:",self._action_space)
        # print("dummy obs:",self._env.get_dummy_observations())
        # print("dummy act:",self._env.get_dummy_actions())

    @property
    def agents(self) -> Sequence[str]:
        """Names of all current agents

        These may be changed as an environment progresses (i.e. agents can be added or removed)
        """
        return self.possible_agents

    @property
    def possible_agents(self) -> Sequence[str]:
        """Names of all possible agents the environment could generate

        These can not be changed as an environment progresses
        """
        return self._env.possible_agents

    @property
    def state_spaces(self) -> gymnasium.Space:
        """State space

        An alias for the ``observation_space`` property
        """
        return {
            uid: space for uid, space in zip(self.possible_agents, [self._observation_space] * self.number_agents)
        }

    @property
    def observation_spaces(self) -> gymnasium.Space:
        """Observation space"""
        return {uid: space for uid, space in zip(self.possible_agents, [self._observation_space] * self.number_agents)}

    @property
    def action_spaces(self) -> gymnasium.Space:
        """Action space"""
        return {uid: space for uid, space in zip(self.possible_agents, [self._action_space] * self.number_agents)}

    def state_space(self,agent):
        """State space for a specific agent"""
        return self.state_spaces[agent]

    def observation_space(self,agent):
        """Observation space for a specific agent"""
        return self.observation_spaces[agent]

    def action_space(self,agent):
        """Action space for a specific agent"""
        return self.action_spaces[agent]

    def _spec_to_space(self, spec: Any) -> gymnasium.Space:
        """Convert the robosuite spec to a Gym space

        :param spec: The robosuite spec to convert
        :type spec: Any supported robosuite spec

        :raises: ValueError if the spec type is not supported

        :return: The Gym space
        :rtype: gymnasium.Space
        """
        if type(spec) is tuple:
            return gymnasium.spaces.Box(shape=spec[0].shape, dtype=np.float32, low=spec[0], high=spec[1])
        elif isinstance(spec, np.ndarray):
            return gymnasium.spaces.Box(
                shape=spec.shape,
                dtype=np.float32,
                low=np.full(spec.shape, float("-inf")),
                high=np.full(spec.shape, float("inf")),
            )
        elif isinstance(spec, torch.Tensor):
            spec.detach().cpu().numpy()
            return gymnasium.spaces.Box(
                shape=spec.shape,
                dtype=np.float32,
                low=np.full(spec.shape, float("-inf")),
                high=np.full(spec.shape, float("inf")),
            )
        elif isinstance(spec, collections.OrderedDict):
            return gymnasium.spaces.Dict({k: self._spec_to_space(v) for k, v in spec.items()})
        else:
            raise ValueError(f"Spec type {type(spec)} not supported. Please report this issue")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        # actions = untensorize_space(self.action_space, unflatten_tensorized_space(self.action_space, actions))
        actions = [actions[uid] for uid in self.possible_agents]
        actions = torch.stack(actions, dim=0)
        observations, rewards, terminated, _ = self._env.step(actions)
        # print("rewards:",rewards)
        # print("terminated",terminated)
        # self._states = states[:, 0]
        self._observations = {uid: observations[i,:] for i, uid in enumerate(self.possible_agents)}
        #.view(-1, 1)
        rewards = {uid: rewards[i,:].view(-1, 1) for i, uid in enumerate(self.possible_agents)}
        terminated = {uid: terminated[i,:].view(-1, 1) for i, uid in enumerate(self.possible_agents)}
        truncated = {uid: torch.zeros_like(value) for uid, value in terminated.items()}
        # print("rewards:",rewards)
        # print("terminated:",terminated)
        return self._observations, rewards, terminated, truncated, self._info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: tuple of dictionaries of torch.Tensor and any other info
        """
        if self._reset_once:
            observations, _ = self._env.reset()
            # self._states = states[:, 0]
            self._observations = {uid: observations[i,:] for i, uid in enumerate(self.possible_agents)}
            self._reset_once = False
        return self._observations, self._info
    
    def state(self):
        # pass
        # self.state_buf = {
        #     f"agent_{agent_id}": self._env.obs_buf[agent_id].reshape(self._env.num_envs, -1)
        #     for agent_id in range(self.number_agents)
        # }
        # # print("state buffer:", self.state_buf)
        # # exit(0)
        return torch.zeros(self._env.num_envs, self._env.num_obs)  # dummy return

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        # self._env.render(*args, **kwargs)
        pass

    def close(self) -> None:
        pass


