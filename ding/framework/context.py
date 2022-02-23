from dataclasses import dataclass
from typing import Any


@dataclass
class Context:
    """
    Overview:
        Context is an object that pass contextual data between middlewares, whose life cycle
        is only one training iteration. It is a dict that reflect itself, so you can set
        any properties as you wish.
    """
    total_step = 0
    train_iter = 0
    env_step = 0
    _kept_keys = set(["total_step", "train_iter", "env_step"])

    def setdefault(self, key: str, default: Any) -> "Context":
        """
        Overview:
            Insert key with a value of default if key is not in the dictionary.
            Return the value for key if key is in the dictionary, else default.
        Arguments:
            - key (:obj:`str`): Key.
            - default (:obj:`Any`): Default value.
        """
        if not hasattr(self, key):
            setattr(self, key, default)
        return self

    def renew(self) -> 'Context':
        """
        Overview:
            Renew context from self, add total_step and shift kept properties to the new instance.
        """
        ctx = type(self)()
        for key in self._kept_keys:
            if hasattr(self, key):
                setattr(ctx, key, getattr(self, key))
        return ctx

    def keep(self, *keys: str) -> "Context":
        """
        Overview:
            Keep this key/keys until next iteration.
        """
        for key in keys:
            self._kept_keys.add(key)
        return self
