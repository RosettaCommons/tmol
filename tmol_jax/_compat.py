"""Install existing module aliases at their public parent package boundary."""

from importlib import import_module
import sys


def alias_modules(package, source, names):
    parent = sys.modules[package]
    for name in names:
        module = import_module(f"{source}.{name}")
        sys.modules[f"{package}.{name}"] = module
        setattr(parent, name, module)
