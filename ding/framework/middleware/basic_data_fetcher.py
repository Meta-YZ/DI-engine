from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
from ding.policy import Policy
from ding.data import Buffer, Dataset, DataLoader

if TYPE_CHECKING:
    from ding.framework import Task, Context


def offpolicy_data_fetcher(task: "Task", cfg: EasyDict, buffer_: Buffer) -> Callable:
    logger = task.logger

    def _fetch(ctx: "Context"):
        try:
            # TODO trajectory sample strategy
            buffered_data = buffer_.sample(cfg.policy.learn.batch_size)
        except ValueError:
            logger.warning(
                "Replay buffer's data is not enough to support training." +
                "You can modify data collect config, e.g. increasing n_sample, n_episode."
            )
            # TODO
            return
        ctx.data = [d.data for d in buffered_data]
        yield
        buffer_.update(ctx.train_output)  # such as priority

    return _fetch


# TODO move ppo training for loop to new middleware
def onpolicy_data_fetcher(task: "Task", cfg: EasyDict, buffer_: Buffer) -> Callable:

    def _fetch(ctx: "Context"):
        buffered_data = buffer_.acquire_all()
        ctx.data = [d.data for d in buffered_data]
        yield
        buffer_.update(ctx.train_output)  # such as priority

    return _fetch


def offline_data_fetcher(task: "Task", cfg: EasyDict, dataset: Dataset) -> Callable:
    dataloader = DataLoader(dataset, batch_size=cfg.policy.learn.batch_size)

    def _fetch(ctx: "Context"):
        buffered_data = next(dataloader)
        ctx.data = [d.data for d in buffered_data]
        # TODO apply update in offline setting when necessary

    return _fetch


# ################ Algorithm-Specific ###########################


def sqil_data_fetcher(task: "Task", cfg: EasyDict, agent_buffer: Buffer, expert_buffer: Buffer) -> Callable:
    logger = task.logger

    def _fetch(ctx: "Context"):
        try:
            agent_buffered_data = agent_buffer.sample(cfg.policy.learn.batch_size // 2)
            expert_buffered_data = expert_buffer.sample(cfg.policy.learn.batch_size // 2)
        except ValueError:
            logger.warning(
                "Replay buffer's data is not enough to support training." +
                "You can modify data collect config, e.g. increasing n_sample, n_episode."
            )
            # TODO
            return
        agent_data = [d.data for d in agent_buffered_data]
        expert_data = [d.data for d in expert_buffered_data]
        ctx.data = agent_data + expert_data

    return _fetch
