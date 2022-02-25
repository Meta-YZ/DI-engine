from typing import TYPE_CHECKING, Callable
import torch
import numpy as np
from easydict import EasyDict
from ding.envs import BaseEnvManager
from ding.policy import Policy
from ding.data import Dataset, DataLoader
from .eval_utils import VectorEvalMonitor, IMetric

if TYPE_CHECKING:
    from ding.framework import Task, Context


def interaction_evaluator(task: "Task", cfg: EasyDict, policy: Policy, env: BaseEnvManager) -> Callable:
    env.seed(cfg.seed, dynamic_seed=False)
    policy = policy.eval_mode
    logger = task.logger

    def _evaluate(ctx: "Context"):
        ctx.setdefault("train_iter", 0)
        ctx.setdefault("last_eval_iter", -1)
        ctx.setdefault("eval_output", None)
        ctx.keep("train_iter", "last_eval_iter", 'eval_output')
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
        logger.info('Current Evaluation: Train Iter({})\tEval Reward({:.3f})'.format(ctx.train_iter, eval_reward))
        ctx.last_eval_iter = ctx.train_iter
        ctx.eval_value = episode_reward

        if stop_flag:
            task.finish = True

    return _evaluate


def metric_evaluator(task: "Task", cfg: EasyDict, policy: Policy, dataset: Dataset, metric: IMetric) -> Callable:
    policy = policy.eval_mode
    dataloader = DataLoader(dataset, batch_size=cfg.policy.eval.batch_size)
    logger = task.logger

    def _evaluate(ctx: "Context"):
        ctx.setdefault("train_iter", 0)
        ctx.setdefault("last_eval_iter", -1)
        ctx.setdefault("eval_output", None)
        ctx.keep("train_iter", "last_eval_iter", 'eval_output')
        if ctx.train_iter == ctx.last_eval_iter or \
            ((ctx.train_iter - ctx.last_eval_iter) <
                cfg.policy.eval.evaluator.eval_freq and ctx.train_iter != 0):
            return

        policy.reset()
        eval_output = []

        for batch_idx, batch_data in enumerate(dataloader):
            inputs, label = batch_data
            policy_output = policy.forward(inputs)
            eval_output.append(metric.eval(policy_output, label))
        # TODO reduce avg_eval_output among different gpus
        avg_eval_output = metric.reduce_mean(eval_output)
        stop_flag = metric.gt(avg_eval_output, cfg.env.stop_value) and ctx.train_iter > 0
        logger.info('Current Evaluation: Train Iter({})\tEval Metric({:.3f})'.format(ctx.train_iter, avg_eval_output))
        ctx.last_eval_iter = ctx.train_iter
        ctx.eval_value = avg_eval_output

        if stop_flag:
            task.finish = True

    return _evaluate


# TODO battle evaluator
