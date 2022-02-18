from typing import TYPE_CHECKING, Callable
import torch
import numpy as np
import logging
from rich import print
from easydict import EasyDict
from ding.envs import BaseEnvManager
from ding.policy import Policy
from ding.data import Buffer
from .eval_utils import VectorEvalMonitor, IMetric

if TYPE_CHECKING:
    from ding.framework import Task, Context


# TODO whether use cfg
def interaction_evaluator(task: "Task", cfg: EasyDict, policy: Policy, env: BaseEnvManager) -> Callable:
    env.seed(cfg.seed, dynamic_seed=False)
    policy = policy.eval_mode

    def _evaluate(ctx: "Context"):

        ctx.setdefault("train_iter", 0)
        ctx.setdefault("last_eval_iter", -1)
        ctx.keep("train_iter", "last_eval_iter")
        if ctx.train_iter == ctx.last_eval_iter or \
            ((ctx.train_iter - ctx.last_eval_iter) <
                cfg.policy.eval.evaluator.eval_freq and ctx.train_iter != 0):
            return

        if env.closed:
            env.launch()
        else:
            env.reset()
        policy.reset()
        eval_monitor = VectorEvalMonitor(env.env_num, cfg.env.n_evaluator_episode)

        while not eval_monitor.is_finished():
            obs = env.ready_obs.tensor(dtype=torch.float32)
            policy_output = policy.forward(obs)
            action = policy_output.action.numpy()
            timesteps = env.step(action).tensor(dtype=torch.float32)
            for env_id, timestep in timesteps.items():
                if timestep.done:
                    policy.reset([env_id])
                    reward = timestep.info.final_eval_reward
                    eval_monitor.update_reward(env_id, reward)
        episode_reward = eval_monitor.get_episode_reward()
        eval_reward = np.mean(episode_reward)
        stop_flag = eval_reward >= cfg.env.stop_value and ctx.train_iter > 0
        # TODO save_ckpt_fn
        ctx.eval_reward = eval_reward
        # TODO evaluator log
        print('Current Evaluation: Train Iter({})\tEval Reward({:.3f})'.format(ctx.train_iter, eval_reward))
        ctx.last_eval_iter = ctx.train_iter
        if stop_flag:
            task.finish = True

    return _evaluate


def metric_evaluator(task: "Task", cfg: EasyDict, policy: Policy, buffer_: Buffer, metric: IMetric) -> Callable:
    policy = policy.eval_mode

    def _evaluate(ctx: "Context"):
        ctx.setdefault("train_iter", 0)
        ctx.setdefault("last_eval_iter", -1)
        ctx.keep("train_iter", "last_eval_iter")
        if ctx.train_iter == ctx.last_eval_iter or \
            ((ctx.train_iter - ctx.last_eval_iter) <
                cfg.policy.eval.evaluator.eval_freq and ctx.train_iter != 0):
            return

        policy.reset()
        buffer_.epoch_prepare(cfg.policy.eval.batch_size)
        eval_results = []

        for batch_idx, batch_data in enumerate(buffer_):
            inputs, label = batch_data
            policy_output = policy.forward(inputs)
            eval_results.append(metric.eval(policy_output, label))
        avg_eval_result = metric.reduce_mean(eval_results)
        # TODO reduce avg_eval_result among different gpus
        stop_flag = metric.gt(avg_eval_result, cfg.env.stop_value) and ctx.train_iter > 0
        print('Current Evaluation: Train Iter({})\tEval Metric({:.3f})'.format(ctx.train_iter, avg_eval_result))
        ctx.last_eval_iter = ctx.train_iter
        if stop_flag:
            task.finish = True

    return _evaluate


# TODO battle evaluator
