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
"""Experiment utils."""

import os

import tensorflow as tf

from cen import losses
from cen import metrics
from cen import models
from cen import networks


class ModeKeys(object):
    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"


def get_input_dtypes(data):
    """Returns input shapes."""
    return {k: str(v.dtype) for k, v in data[0].items()}


def get_input_shapes(data):
    """Returns input shapes."""
    return {k: v.shape[1:] for k, v in data[0].items()}


def get_output_shape(data):
    """Returns output shapes."""
    return data[1].shape[1:]


def build(
    input_dtypes,
    input_shapes,
    output_shape,
    mode=ModeKeys.TRAIN,
    working_dir=None,
    model_name=None,
    model_kwargs=None,
    network_kwargs=None,
    loss_kwargs=None,
    optimizer_kwargs=None,
    checkpoint_kwargs=None,
    tensorboard_kwargs=None,
    eval_metrics=None,

):
    """
    Builds model and callbacks for training or evaluation.

    Parameters
    ----------
    input_dtypes : dict of {str : str}
        Datatypes of the low level context features "C" and the high level
        features "X".
    input_shapes : dict of {str : tuple of (int,)}
        Shape tuples (not including the batch size) of the low level context
        features "C" and the high level features "X".
    output_shape : tuple of (int,)
        Target shape of the model outputs.
    mode : str, optional
        One of "train", "eval", and "infer", by default "train".
        Note: "eval" and "infer" are not yet supported.
    working_dir : str, optional
        Directory to use for, e.g., saving model checkpoints while training,
        by default None.
    model_name : str, optional
        The type of model to build. One of "baseline", "cen", and "moe",
        by default None.
    model_kwargs : dict, optional
        Keyword arguments for the method `cen.models.get`, by default None.
    network_kwargs : dict, optional
        Keyword arguments for the method `cen.networks.get`, by default None.
    loss_kwargs : dict, optional
        Keyword arguments for the method `cen.losses.get`, by default None.
    optimizer_kwargs : dict, optional
        Keyword arguments for the method `tensorflow.keras.optimizers.get`,
        by default None.
    checkpoint_kwargs : dict, optional
        Keyword arguments for the method
        `tensorflow.keras.callbacks.ModelCheckpoint`, by default None.
    tensorboard_kwargs : dict, optional
        Keyword arguments for the method
        `tensorflow.keras.callbacks.TensorBoard`, by default None.
    eval_metrics : dict of {str : dict}, optional
        Mapping of user selected labels to keyword arguments for the method
        `cen.metrics.get`, by default None.

    Returns
    -------
    tensorflow.keras.models.Model
        The untrained model.
    dict
        Information about the model's "context" layer and "encodings",
        if applicable.
    """

    tf.keras.backend.clear_session()
    if working_dir is None:
        working_dir = os.getcwd()

    model_kwargs = model_kwargs if model_kwargs else {}
    network_kwargs = network_kwargs if network_kwargs else {}
    loss_kwargs = loss_kwargs if loss_kwargs else {}
    optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
    checkpoint_kwargs = checkpoint_kwargs if checkpoint_kwargs else {}
    tensorboard_kwargs = tensorboard_kwargs if tensorboard_kwargs else {}

    # Build model.
    net = networks.get(**network_kwargs)
    model, info = models.get(
        model_name,
        encoder=net,
        input_dtypes=input_dtypes,
        input_shapes=input_shapes,
        output_shape=output_shape,
        **model_kwargs,
    )

    # Build loss and optimizer.
    loss = losses.get(**loss_kwargs)
    opt = tf.keras.optimizers.get(dict(**optimizer_kwargs))

    # Build metrics.
    metrics_list = None
    if eval_metrics:
        metrics_list = [metrics.get(**v) for _, v in eval_metrics.items()]

    # Compile model for training.
    if mode == ModeKeys.TRAIN:
        model.compile(optimizer=opt, loss=loss, metrics=metrics_list)
        callbacks = []
        if checkpoint_kwargs:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(working_dir, "checkpoint"),
                    **checkpoint_kwargs,
                )
            )
        if tensorboard_kwargs:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(working_dir, "tensorboard"),
                    **tensorboard_kwargs,
                )
            )
        info["callbacks"] = callbacks
        return model, info

    # Compile model for evaluation or inference.
    else:
        model.compile(loss=loss, optimizer=opt, metrics=metrics_list)
        checkpoint_path = os.path.join(working_dir, "checkpoint")
        model.load_weights(checkpoint_path).expect_partial()
        return model, info
