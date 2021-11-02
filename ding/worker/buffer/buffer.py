from typing import Any, Callable, List, Optional, Tuple, Union
from ding.worker.buffer.storage import Storage
import copy


def apply_middleware(func_name: str):

    def wrap_func(base_func: Callable):

        def handler(buffer, *args, **kwargs):
            """
            Overview:
                The real processing starts here, we apply the middleware one by one,
                each middleware will receive next `chained` function, which is an executor of next
                middleware. You can change the input arguments to the next `chained` middleware, and you
                also can get the return value from the next middleware, so you have the
                maximum freedom to choose at what stage to implement your method.
            """

            def wrap_handler(middleware, *args, **kwargs):
                if len(middleware) == 0:
                    return base_func(buffer, *args, **kwargs)

                def chain(*args, **kwargs):
                    return wrap_handler(middleware[1:], *args, **kwargs)

                func = middleware[0]
                return func(func_name, chain, *args, **kwargs)

            return wrap_handler(buffer.middleware, *args, **kwargs)

        return handler

    return wrap_func


class Buffer:

    def __init__(self, storage: Storage) -> None:
        """
        Overview:
            Initialize the buffer
        Arguments:
            - storage (:obj:`Storage`): The storage instance.
        """
        self.storage = storage
        self.middleware = []

    @apply_middleware("push")
    def push(self, data: Any, meta: Optional[dict] = None) -> None:
        """
        Overview:
            Push data and it's meta information in buffer.
        Arguments:
            - data (:obj:`Any`): The data which will be pushed into buffer.
            - meta (:obj:`dict`): Meta information, e.g. priority, count, staleness.
        """
        self.storage.append(data, meta)

    @apply_middleware("sample")
    def sample(
            self,
            size: int,
            replace: bool = False,
            range: slice = None,
            return_index: bool = False,
            return_meta: bool = False
    ) -> List[Union[Any, Tuple[Any, str], Tuple[Any, str, dict]]]:
        """
        Overview:
            Sample data with length ``size``, this function may be wrapped by middleware.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - replace (:obj:`bool`): If use replace is true, you may receive duplicated data from the buffer.
            - range (:obj:`slice`): Range slice.
            - return_index (:obj:`bool`): Transform the return value to (data, index),
            - return_meta (:obj:`bool`): Transform the return value to (data, meta),
                or (data, index, meta) if return_index is true.
        Returns:
            - sample_data (:obj:`list`): A list of data with length ``size``.
        """
        return self.storage.sample(
            size, replace=replace, range=range, return_index=return_index, return_meta=return_meta
        )

    @apply_middleware("clear")
    def clear(self) -> None:
        """
        Overview:
            Clear the storage
        """
        self.storage.clear()

    @apply_middleware("update")
    def update(self, index: str, data: Any, meta: dict) -> bool:
        """
        Overview:
            Update data and meta by index
        Arguments:
            - index (:obj:`str`): Index of data.
            - data (:obj:`any`): Pure data.
            - meta (:obj:`dict`): Meta information.
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """
        return self.storage.update(index, data, meta)

    @apply_middleware("delete")
    def delete(self, index: str) -> bool:
        """
        Overview:
            Delete one data sample by index
        Arguments:
            - index (:obj:`str`): Index
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """
        return self.storage.delete(index)

    def use(self, func: Callable) -> "Buffer":
        r"""
        Overview:
            Use algorithm middleware to modify the behavior of the buffer.
            Every middleware should be a callable function, it will receive three argument parts, including:
            1. The buffer instance, you can use this instance to visit every thing of the buffer,
               including the storage.
            2. The functions called by the user, there are three methods named `push`, `sample` and `clear`,
               so you can use these function name to decide which action to choose.
            3. The remaining arguments passed by the user to the original function, will be passed in *args.

            Each middleware handler should return two parts of the value, including:
            1. The first value is `done` (True or False), if done==True, the middleware chain will stop immediately,
               no more middleware will be executed during this execution
            2. The remaining values, will be passed to the next middleware or the default function in the buffer.
        Arguments:
            - func (:obj:`Callable`): The middleware handler
        Returns:
            - buffer (:obj:`Buffer`): The instance self
        """
        self.middleware.append(func)
        return self

    def view(self) -> "Buffer":
        r"""
        Overview:
            A view is a new instance of buffer, with a deepcopy of every property except the storage.
            The storage is shared among all the buffer instances.
        Returns:
            - buffer (:obj:`Buffer`): The instance self
        """
        buffer = Buffer(self.storage)
        buffer.middleware = copy.deepcopy(self.middleware)
        return buffer

    def count(self) -> int:
        """
        Overview:
            Return the actual valid data count in buffer now
        Returns:
            - count (:obj:`int`): The actual valid data count
        """
        return self.storage.count()