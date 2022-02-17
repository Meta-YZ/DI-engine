from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import treetensor.torch as ttorch
from ding.buffer import Buffer
from ding.policy import Policy
if TYPE_CHECKING:
    from ding.framework import Task, Context


def policy_enhance(cfg: EasyDict, buffer_: Buffer, policy: Policy) -> Callable:
    """
    For example: gae, nstep
    """

    def _enhance(ctx: "Context"):
        # TODO whether add a uniform interface for buffer sample
        gae_data = policy.get_train_sample(buffer_)
        buffer_.next_data = gae_data

    return _enhance


def sqil(cfg: EasyDict, agent_buffer: Buffer, expert_buffer: Buffer) -> Callable:

    def _enhance(ctx: "Context"):
        bs = cfg.policy.learn.batch_size
        agent_buffered_data = agent_buffer.sample(bs // 2)
        expert_buffered_data = expert_buffer.sample(bs // 2)

        # TODO check nstep+sqil
        agent_buffered_data.data.reward = ttorch.zeros_like(agent_buffered_data.data.reward)
        expert_buffered_data.data.reward = ttorch.ones_like(expert_buffered_data.data.reward)

        agent_buffer.next_data = agent_buffered_data
        expert_buffer.next_data = expert_buffered_data

    return _enhance


def rnd(cfg: EasyDict, buffer_: Buffer, reward_model):

    def _enhance(ctx: "Context"):
        buffered_data = buffer_.sample(cfg.policy.learn.batch_size)
        buffered_data.data = reward_model.estimate(buffered_data.data)
        buffer_.next_data = buffered_data

    return _enhance


# TODO MBPO
# TODO SIL
# TODO TD3 VAE
