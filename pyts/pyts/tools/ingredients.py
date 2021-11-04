import os

from sacred import Ingredient

from .builders import (
    CartesianBuilder
)
from .loaders import TSLoader
from .loggers import SacredMetricLogger
from .radials import get_radial_factory


# ===== Dataset Ingredient(s) ===== #
data_ingredient = Ingredient("data_loader")


@data_ingredient.capture
def get_data_loader(
    loader_type: str = "ts_loader", **kwargs,
):
    """
    :param loader_type: str. Defaults to 'ts_loader'. Used to specify which loader type is
        being used. Supported identifiers: 'ts_loader'
    :param kwargs: kwargs passed directly to Loader classes
    :return: DataLoader object specified by `loader_type`
    """
    if loader_type == "ts_loader":
    else:
        raise ValueError(
            "arg `loader_type` had value: {} which is not supported. "
            "Check ingredient docs for supported strings "
            "identifiers".format(loader_type)
        )


# ===== Builder Ingredient(s) ===== #
builder_ingredient = Ingredient("model_builder")


@builder_ingredient.capture
def get_builder(
    builder_type: str = "cartesian_builder", **kwargs,
):
    """

    :param builder_type: str. Defaults to 'cartesian_builder'.
    :param kwargs: kwargs passed directly to Builder classes
    :return: Builder object specified by 'builder_type'
    """
    kwargs["radial_factory"] = get_radial_factory(
        kwargs.get("radial_factory", "multi_dense"), kwargs.get("radial_kwargs", None)
    )

    if builder_type == "cartesian_builder":
        return CartesianBuilder(**kwargs)
    else:
        raise ValueError(
            "arg `builder_type` had value: {} which is not supported. Check "
            "ingredient docs for supported string identifiers".format(builder_type)
        )


# ===== Logger Ingredient(s) ===== #
logger_ingredient = Ingredient("metric_logger")
get_logger = logger_ingredient.capture(SacredMetricLogger)
