"""Universal DeMark engine embedded in Sora Watermark Cleaner.

Derived from linkedlist771/DeMark-World under the MIT License. The macOS
runtime, dependency loading, and compatibility surface were modified here.
"""

__all__ = ["CleanerType", "DeMarkWorld"]


def __getattr__(name: str):
    if name == "DeMarkWorld":
        from demark_world.core import DeMarkWorld

        return DeMarkWorld
    if name == "CleanerType":
        from demark_world.schemas import CleanerType

        return CleanerType
    raise AttributeError(name)
