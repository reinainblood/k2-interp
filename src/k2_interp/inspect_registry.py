"""Inspect AI entry point: registers the ``k2interp`` model provider.

Loaded through ``[project.entry-points.inspect_ai]`` so ``--model k2interp/<checkpoint>``
works without importing anything by hand. The provider module (and inspect_ai) is
imported only when Inspect actually resolves the provider.
"""

from inspect_ai.model import modelapi


@modelapi(name="k2interp")
def k2interp():
    from .inspect_provider import K2InterpAPI

    return K2InterpAPI
