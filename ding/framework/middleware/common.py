from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import numpy as np

from ding.utils import save_file
from ding.policy import Policy

if TYPE_CHECKING:
    from ding.framework import Task, Context


def ckpt_saver(task: "Task", cfg: EasyDict, policy: Policy, train_freq: int = None) -> Callable:
    logger = task.logger

    def _save(ctx: "Context"):
        if train_freq:
            ctx.setdefault("last_save_iter", 0)
            ctx.keep("last_save_iter")
        ctx.setdefault("max_eval_value", -np.inf)
        ctx.keep("max_eval_value")
        # train enough iteration
        if train_freq and ctx.train_iter - ctx.last_save_iter >= train_freq:
            save_file(
                "{}/iteration_{}.pth.tar".format(task.instance_name, ctx.train_iter), policy.learn_mode.state_dict()
            )
            ctx.last_save_iter = ctx.train_iter

        # best eval reward so far
        if ctx.eval_value > ctx.max_eval_value:
            save_file("{}/eval.pth.tar".format(task.instance_name), policy.learn_mode.state_dict())
            ctx.max_eval_value = ctx.eval_value

        # finish
        if task.finish:
            save_file("{}/final.pth.tar".format(task.instance_name), policy.learn_mode.state_dict())

    return _save
