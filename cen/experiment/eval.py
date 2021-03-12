#  Copyright 2020 Maruan Al-Shedivat. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================
"""Evaluation."""

import logging

from cen.experiment import utils

logger = logging.getLogger(__name__)


def evaluate(
    datasets,
    eval_metrics,
    build_kwargs=None,
    model=None,
    batch_size=None,
    verbose=1
):
    """
    Evaluates performance of a provided model, or a model built using provided
    parameters, using the specified metrics.

    Parameters
    ----------
    datasets : dict { str : tuple of np.ndarray }
        Tuples (X, y) mapped to names "train", "valid", and "test".
    eval_metrics : dict of {str : dict}
        Mapping of user selected labels to keyword arguments for the method
        `cen.metrics.get`, by default None.
    build_kwargs : dict, optional
        Keyword arguments for `cen.experiment.utils.build`, used to build a
        model if one is not provided, by default None.
    model : tensorflow.keras.models.Model, optional
        Pre-trained model to evaluate, by default None.
    batch_size : int, optional
        `batch_size` parameter for `tensorflow.keras.models.Model.evaluate`,
        by default None.
    verbose : int, optional
        Sets verbosity level during TensorFlow evaluation, by default 1.

    Returns
    -------
    dict { str: dict { str: float } }
        Mapping of metric names to their values, for each of the "train",
        "valid", and "test" datasets.
    """
    build_kwargs = build_kwargs if build_kwargs else {}

    if model is None:
        logger.info("Building...")

        input_dtypes = utils.get_input_dtypes(datasets["test"])
        input_shapes = utils.get_input_shapes(datasets["test"])
        output_shape = utils.get_output_shape(datasets["test"])
        if model is None:
            model = utils.build(
                input_dtypes=input_dtypes,
                input_shapes=input_shapes,
                output_shape=output_shape,
                mode=utils.ModeKeys.EVAL,
                **build_kwargs
            )

    logger.info("Evaluating...")

    metrics = {}
    for set_name, dataset in datasets.items():
        if dataset is None or dataset[1] is None:
            continue
        metric_names = ["loss"] + list(eval_metrics.keys())
        metric_values = model.evaluate(
            *dataset, batch_size=batch_size, verbose=verbose
        )
        metrics[set_name] = dict(zip(metric_names, metric_values))
        logger.info(f"{set_name} metrics: {metrics[set_name]}")

    return metrics
