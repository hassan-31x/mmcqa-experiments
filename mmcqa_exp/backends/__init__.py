"""Backends are imported lazily: the Pangea worker runs with transformers 4.37 and the Qwen
worker with transformers 5.x, so neither module may import the other's dependencies."""


def get_backend(name: str):
    if name == "qwen":
        from .qwen import QwenBackend
        return QwenBackend
    if name == "qwen9b":
        from .qwen import Qwen9BBackend
        return Qwen9BBackend
    if name == "pangea":
        from .pangea import PangeaBackend
        return PangeaBackend
    if name == "fake":
        from .fake import FakeBackend
        return FakeBackend
    raise ValueError(f"Unknown backend: {name!r}")
