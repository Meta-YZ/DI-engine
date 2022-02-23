from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import treetensor.torch as ttorch
from ding.data import Buffer
from ding.policy import Policy
if TYPE_CHECKING:
    from ding.framework import Task, Context


def enhance_from_trajectory_view(cfg: EasyDict, policy: Policy) -> Callable:
    """
    Usage: Before push info buffer or sample by trajectory
    For example: gae, nstep
    """
    policy = policy.collect_mode

    def _enhance(ctx: "Context"):
        ctx.data = policy.get_train_sample(ctx.data)

    return _enhance


def rnd(cfg: EasyDict, reward_policy: Policy) -> Callable:
    reward_policy = reward_policy.eval_mode

    def _enhance(ctx: "Context"):
        ctx.data = reward_policy.forward(ctx.data)

    return _enhance


# TODO nstep reward
# TODO MBPO
# TODO SIL
# TODO SQIL
# TODO TD3 VAE
