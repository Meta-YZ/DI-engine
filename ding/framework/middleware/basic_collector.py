from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import torch
from ding.envs import BaseEnvManager
from ding.policy import Policy
from ding.data import Buffer

if TYPE_CHECKING:
    from ding.framework import Task, Context


# TODO ctx member variable definition
def inferencer(task: "Task", cfg: EasyDict, policy: Policy, env: BaseEnvManager) -> Callable:
    env.seed(cfg.seed)
    policy = policy.collect_mode

    def _inference(ctx: "Context"):
        if env.closed:
            env.launch()
        ctx.setdefault("env_step", 0)
        ctx.keep("env_step")

        obs = env.ready_obs
        # TODO mask necessary rollout
        policy_kwargs = {}
        # policy_kwargs = {'eps': eps_greedy(ctx.env_step)}
        # TODO eps greedy

        policy_output = policy.forward(obs, **policy_kwargs)
        ctx.action = policy_output.action.numpy()
        ctx.policy_output = policy_output

    return _inference


def rolloutor(task: "Task", cfg: EasyDict, policy: Policy, env: BaseEnvManager, buffer_: Buffer) -> Callable:
    # TODO whether need to access member variable
    policy = policy.collect_mode

    def _rollout(ctx):
        ctx.setdefault("env_episode", 0)
        ctx.keep("env_episode")
        timesteps = env.step(ctx.action)
        ctx.env_step += len(timesteps)
        timesteps = timesteps.tensor(dtype=torch.float32)
        transitions = policy.process_transition(ctx.obs, ctx.policy_output, timesteps)
        transitions.collect_train_iter = ctx.train_iter
        buffer_.push(transitions)
        # TODO abnormal env step
        for env_id, timestep in timesteps.items():
            if timestep.done:
                policy.reset([env_id])
                ctx.env_episode += 1
        # TODO log

    return _rollout


def step_collector(task: "Task", cfg: EasyDict, policy: Policy, env: BaseEnvManager, buffer_: Buffer) -> Callable:
    _inferencer = inferencer(task, cfg, policy, env)
    _rolloutor = rolloutor(task, cfg, policy, env, buffer_)

    def _collect(ctx: "Context"):
        old = ctx.env_step
        while True:
            _inferencer(ctx)
            _rolloutor(ctx)
            if ctx.env_step - old > cfg.policy.collect.n_sample * cfg.policy.collect.unroll_len:
                break

    return _collect


def episode_collector(task: "Task", cfg: EasyDict, policy: Policy, env: BaseEnvManager, buffer_: Buffer) -> Callable:
    _inferencer = inferencer(task, cfg, policy, env)
    _rolloutor = rolloutor(task, cfg, policy, env, buffer_)

    def _collect(ctx: "Context"):
        old = ctx.env_episode
        while True:
            _inferencer(ctx)
            _rolloutor(ctx)
            if ctx.env_episode - old > cfg.policy.collect.n_episode:
                break

    return _collect


# TODO battle collector
