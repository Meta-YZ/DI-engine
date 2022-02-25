from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import torch
from ding.envs import BaseEnvManager
from ding.policy import Policy
from ding.data import Buffer

if TYPE_CHECKING:
    from ding.framework import Task, Context


from ding.rl_utils import get_eps_greedy_fn


def eps_greedy(task: "Task", cfg: EasyDict) -> Callable:
    handle = get_eps_greedy_fn(cfg.policy.other.eps)

    def _eps_greedy(ctx: "Context"):
        ctx.collect_kwargs['eps'] = handle(ctx.env_step)
        yield
        try:
            ctx.collect_kwargs.pop('eps')
        except:  # noqa
            pass

    return _eps_greedy


def inferencer(task: "Task", cfg: EasyDict, policy: Policy, env: BaseEnvManager) -> Callable:
    env.seed(cfg.seed)
    policy = policy.collect_mode

    def _inference(ctx: "Context"):
        if env.closed:
            env.launch()
        ctx.setdefault("env_step", 0)
        ctx.setdefault("collect_kwargs", {})
        ctx.keep("env_step")

        obs = env.ready_obs
        # TODO mask necessary rollout

        inference_output = policy.forward(obs, **ctx.collect_kwargs)
        ctx.action = inference_output.action.numpy()
        ctx.inference_output = inference_output

    return _inference


def rolloutor(task: "Task", cfg: EasyDict, policy: Policy, env: BaseEnvManager, buffer_: Buffer) -> Callable:
    policy = policy.collect_mode
    logger = task.logger

    def _rollout(ctx):
        ctx.setdefault("env_episode", 0)
        ctx.keep("env_episode")
        timesteps = env.step(ctx.action)
        ctx.env_step += len(timesteps)
        timesteps = timesteps.tensor(dtype=torch.float32)
        transitions = policy.process_transition(ctx.obs, ctx.inference_output, timesteps)
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
