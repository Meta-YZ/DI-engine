from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
from ding.policy import Policy

if TYPE_CHECKING:
    from ding.framework import Task, Context


def trainer(task: "Task", cfg: EasyDict, policy: Policy) -> Callable:
    policy = policy.learn_mode
    logger = task.logger

    def _train(ctx: "Context"):
        ctx.setdefault("train_iter", 0)
        ctx.keep("train_iter")
        ctx.setdefault("train_output", {})
        ctx.keep("train_output")

        if ctx.data is None:  # no enough data from data fetcher
            return
        train_output = policy.forward(ctx.data)
        if ctx.train_iter % cfg.train_log_freq == 0:
            logger.info(
                'Current Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, train_output['total_loss'])
            )
        ctx.train_iter += 1
        ctx.train_output = train_output

    return _train


# TODO reward model
