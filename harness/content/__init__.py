from .base import ContentSet, Component, ItemFamily, Item, Distractor  # noqa: F401
from .demo import DemoContentSet  # noqa: F401
from .finance import RContentSet, SContentSet  # noqa: F401
from .novel import YContentSet  # noqa: F401

REGISTRY = {"demo": DemoContentSet, "R": RContentSet, "S": SContentSet, "Y": YContentSet}


def load_content(name: str) -> ContentSet:
    """Return a content set by name. R, S and Y are registered here once
    their frozen specifications exist (Gate 0)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown content set {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]()
