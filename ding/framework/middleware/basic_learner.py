from typing import TYPE_CHECKING, Callable
import logging
from rich import print
from easydict import EasyDict
from ding.policy import Policy
from ding.buffer import Buffer

if TYPE_CHECKING:
    from ding.framework import Task, Context


def offpolicy_learner(task: "Task", cfg: EasyDict, policy: Policy, buffer_: Buffer) -> Callable:
    policy = policy.learn_mode

    def _learn(ctx: "Context"):
        ctx.setdefault("train_iter", 0)
        ctx.keep("train_iter")

        for i in range(policy.instance_cfg.learn.update_per_collect):
            try:
                buffered_data = buffer_.sample(policy.instance_cfg.learn.batch_size)
            except ValueError:
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            data = [d.data for d in buffered_data]
            learn_output = policy.forward(data)
            # TODO learner log
            if ctx.train_iter % 20 == 0:
                print(
                    'Current Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, learn_output['total_loss'])
                )
            ctx.train_iter += 1

    return _learn


online_learner = offpolicy_learner


def onpolicy_learner(task: "Task", cfg: EasyDict, policy: Policy, buffer_: Buffer) -> Callable:
    policy = policy.learn_mode

    def _learn(ctx: "Context"):
        ctx.setdefault("train_iter", 0)
        ctx.keep("train_iter")

        buffered_data = buffer_.acquire_all()
        data = buffered_data.data
        learn_output = policy.forward(data)
        print(
            'Current Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, learn_output['total_loss'])
        )
        ctx.train_iter += learn_output["update_iter_count"]

    return _learn


def offline_learner(task: "Task", cfg: EasyDict, policy: Policy, buffer_: Buffer) -> Callable:
    """
    .. note::
        This middleware can be applied in both offline RL and imitation learning.
    """
    policy = policy.learn_mode

    def _learn(ctx: "Context"):
        ctx.setdefault("train_iter", 0)
        ctx.keep("train_iter")

        buffered_data = buffer_.next_batch()
        data = buffered_data.data
        learn_output = policy.forward(data)
        if ctx.train_iter % 20 == 0:
            print(
                'Current Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, learn_output['total_loss'])
            )
        ctx.train_iter += 1

    return _learn

# TODO priority
# TODO reward model
# TODO SQIL
