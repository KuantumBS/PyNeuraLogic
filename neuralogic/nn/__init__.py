from typing import Optional

from neuralogic.core.enums import Backend


def get_neuralogic_layer(backend: Backend = Backend.JAVA):
    if backend == Backend.DYNET:
        from neuralogic.nn.dynet import NeuraLogic  # type: ignore

        return NeuraLogic
    if backend == Backend.JAVA:
        from neuralogic.nn.java import NeuraLogic  # type: ignore

        return NeuraLogic
    if backend == Backend.TORCH:
        from neuralogic.nn.torch import NeuraLogic  # type: ignore

        return NeuraLogic
    raise NotImplementedError


def get_evaluator(
    template: "Template",
    backend: Backend = Backend.JAVA,
    settings=None,
):
    from neuralogic.core.settings import Settings

    if settings is None:
        settings = Settings()

    if backend == Backend.DYNET:
        from neuralogic.nn.evaluators.dynet import DynetEvaluator

        return DynetEvaluator(template, settings)
    if backend == Backend.JAVA:
        from neuralogic.nn.evaluators.java import JavaEvaluator

        return JavaEvaluator(template, settings)
    if backend == Backend.TORCH:
        from neuralogic.nn.evaluators.torch import TorchEvaluator

        return TorchEvaluator(template, settings)
