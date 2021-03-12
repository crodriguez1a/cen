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
"""Supervised training."""

import logging
import os

import tensorflow as tf

from cen.experiment import utils

logger = logging.getLogger(__name__)


def train(
    train_data,
    shuffle_buffer_size,
    batch_size,
    epochs=1,
    seed=None,
    build_kwargs=None,
    validation_data=None,
    checkpoint_kwargs=None,
    verbose=1
):
    """
    Trains a TensorFlow model using the given dataset and params.

    Parameters
    ----------
    train_data : tuple of (np.ndarray, np.ndarray)
        Tuple (X, y) of training data.
    shuffle_buffer_size : int
        The `buffer_size` parameter for the method
        `tensorflow.data.Dataset.shuffle`.
    batch_size : int
        Parameter for the method `tensorflow.data.Dataset.batch`.
    epochs : int, optional
        Number of training epochs, by default 1
    seed : int, optional
        Random seed for the method tensorflow.data.Dataset.shuffle`.
    build_kwargs : dict, optional
        Keyword arguments for `cen.experiment.utils.build`, by default None.
    validation_data : tuple of (np.ndarray, np.ndarray), optional
        Tuple (X, y) of validation data, by default None.
    checkpoint_kwargs : dict, optional
        Keyword arguments for the method
        `tensorflow.keras.callbacks.ModelCheckpoint`, by default None.
    verbose : int, optional
        Sets verbosity level during TensorFlow training, by default 1.

    Returns
    -------
    tensorflow.keras.models.Model
        The trained model.
    dict
        Information about the model's "context" layer and "encodings",
        if applicable.
    """
    logger.info("Building...")

    build_kwargs = build_kwargs if build_kwargs else {}

    # Build the model.
    input_dtypes = utils.get_input_dtypes(train_data)
    input_shapes = utils.get_input_shapes(train_data)
    output_shape = utils.get_output_shape(train_data)
    model, info = utils.build(
        input_dtypes=input_dtypes,
        input_shapes=input_shapes,
        output_shape=output_shape,
        mode=utils.ModeKeys.TRAIN,
        **build_kwargs
    )

    # Build datasets.
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_data)
        .shuffle(
            shuffle_buffer_size,
            reshuffle_each_iteration=True,
            seed=seed,
        )
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    validation_dataset = None
    if validation_data is not None:
        validation_dataset = (
            tf.data.Dataset.from_tensor_slices(validation_data)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    logger.info("Training...")

    history = model.fit(
        train_dataset,
        callbacks=info["callbacks"],
        epochs=epochs,
        validation_data=validation_dataset,
        verbose=verbose,
    )
    info["history"] = history.history

    checkpoint_path = os.path.join(os.getcwd(), "checkpoint")
    if checkpoint_kwargs is None:
        # Save model weights if checkpointing was off.
        model.save_weights(checkpoint_path)
    else:
        # Load the best weights if checkpointing was on.
        model.load_weights(checkpoint_path).expect_partial()

    return model, info
