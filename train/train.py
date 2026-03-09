# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple speech recognition to spot a limited number of keywords.

Refactored to TensorFlow v2, using tf.keras, tf.GradientTape, and
tf.data pipelines in place of tf.compat.v1 Session-based APIs.

Usage:
  python train_tf2.py [--wanted_words=yes,no,up,down ...]

See original script docstring for full data/flag documentation.
"""
import argparse
import logging
import os

import numpy as np
import tensorflow as tf

import input_data
import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FLAGS = None


# ---------------------------------------------------------------------------
# Loss / metric helpers
# ---------------------------------------------------------------------------

def compute_loss_and_accuracy(model, fingerprints, labels, training=False):
    """Run a forward pass and return (loss, accuracy, logits)."""
    logits = model(fingerprints, training=training)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )
    loss = tf.reduce_mean(loss)
    predicted = tf.argmax(logits, axis=1, output_type=tf.int64)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted, tf.cast(labels, tf.int64)), tf.float32)
    )
    return loss, accuracy, logits


def compute_confusion_matrix(model, fingerprints, labels, num_classes):
    logits = model(fingerprints, training=False)
    predicted = tf.argmax(logits, axis=1, output_type=tf.int64)
    return tf.math.confusion_matrix(
        tf.cast(labels, tf.int64), predicted, num_classes=num_classes
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Model / audio settings ----
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(","))),
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.feature_bin_count,
        FLAGS.preprocess,
    )

    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url,
        FLAGS.data_dir,
        FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(","),
        FLAGS.validation_percentage,
        FLAGS.testing_percentage,
        model_settings,
        FLAGS.summaries_dir,
    )

    fingerprint_size = model_settings["fingerprint_size"]
    label_count = model_settings["label_count"]
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

    # ---- Learning-rate schedule ----
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(",")))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(",")))
    if len(training_steps_list) != len(learning_rates_list):
        raise ValueError(
            "--how_many_training_steps and --learning_rate must have equal length, "
            f"got {len(training_steps_list)} vs {len(learning_rates_list)}"
        )

    # ---- Build Keras model ----
    # models.create_model_keras should return a tf.keras.Model.
    # If your models.py still returns a graph-mode model, wrap it:
    #   model = models.create_model_keras(model_settings, FLAGS.model_architecture)
    # For backwards compatibility we build a simple wrapper below.
    inputs = tf.keras.Input(shape=(fingerprint_size,), name="fingerprint_input")
    outputs, _ = models.create_model(
        inputs, model_settings, FLAGS.model_architecture, is_training=True
    )
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # ---- Optimizer ----
    # We'll update the learning rate dynamically, so start with a placeholder value.
    if FLAGS.optimizer == "gradient_descent":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rates_list[0])
    elif FLAGS.optimizer == "momentum":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rates_list[0], momentum=0.9, nesterov=True
        )
    else:
        raise ValueError(f"Invalid optimizer: {FLAGS.optimizer}")

    # ---- Checkpointing ----
    os.makedirs(FLAGS.train_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        directory=FLAGS.train_dir,
        max_to_keep=5,
        checkpoint_name=FLAGS.model_architecture,
    )

    start_step = 1
    if FLAGS.start_checkpoint:
        ckpt.restore(FLAGS.start_checkpoint).expect_partial()
        logger.info("Restored checkpoint from %s", FLAGS.start_checkpoint)
    elif ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        logger.info("Restored latest checkpoint: %s", ckpt_manager.latest_checkpoint)

    # ---- TensorBoard summary writers ----
    train_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.summaries_dir, "train")
    )
    validation_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.summaries_dir, "validation")
    )

    # ---- Save word list ----
    labels_path = os.path.join(
        FLAGS.train_dir, FLAGS.model_architecture + "_labels.txt"
    )
    with open(labels_path, "w") as f:
        f.write("\n".join(audio_processor.words_list))

    # ---- Training loop ----
    training_steps_max = int(np.sum(training_steps_list))

    # We need a "fake" session object for AudioProcessor.get_data which was
    # written for TF1.  Pass None — callers that truly need a session should
    # update input_data.py; most modern versions accept None or a tf.function.
    sess = None  # replace with a real session only if your input_data.py needs it

    for training_step in range(start_step, training_steps_max + 1):
        # Determine learning rate for this step
        steps_sum = 0
        for i, steps in enumerate(training_steps_list):
            steps_sum += steps
            if training_step <= steps_sum:
                lr_value = learning_rates_list[i]
                break
        optimizer.learning_rate.assign(lr_value)

        # Fetch training batch
        train_fingerprints, train_ground_truth = audio_processor.get_data(
            FLAGS.batch_size,
            0,
            model_settings,
            FLAGS.background_frequency,
            FLAGS.background_volume,
            time_shift_samples,
            "training",
            sess,
        )
        train_fingerprints = tf.constant(train_fingerprints, dtype=tf.float32)
        train_ground_truth = tf.constant(train_ground_truth, dtype=tf.int64)

        # Gradient tape forward / backward pass
        with tf.GradientTape() as tape:
            train_loss, train_accuracy, _ = compute_loss_and_accuracy(
                model, train_fingerprints, train_ground_truth, training=True
            )
        gradients = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Check for NaNs if requested
        if FLAGS.check_nans:
            for g in gradients:
                if g is not None and tf.reduce_any(tf.math.is_nan(g)):
                    raise RuntimeError("NaN detected in gradients at step %d" % training_step)

        # Write training summaries
        with train_writer.as_default():
            tf.summary.scalar("cross_entropy", train_loss, step=training_step)
            tf.summary.scalar("accuracy", train_accuracy, step=training_step)

        logger.debug(
            "Step #%d: rate %f, accuracy %.1f%%, cross entropy %f",
            training_step, lr_value, train_accuracy * 100, train_loss,
        )

        is_last_step = training_step == training_steps_max
        if training_step % FLAGS.eval_step_interval == 0 or is_last_step:
            logger.info(
                "Step #%d: rate %f, accuracy %.1f%%, cross entropy %f",
                training_step, lr_value, train_accuracy * 100, train_loss,
            )

            # ---- Validation ----
            set_size = audio_processor.set_size("validation")
            total_accuracy = 0.0
            total_conf_matrix = None

            for offset in range(0, set_size, FLAGS.batch_size):
                val_fingerprints, val_ground_truth = audio_processor.get_data(
                    FLAGS.batch_size,
                    offset,
                    model_settings,
                    0.0,
                    0.0,
                    0,
                    "validation",
                    sess,
                )
                val_fingerprints = tf.constant(val_fingerprints, dtype=tf.float32)
                val_ground_truth = tf.constant(val_ground_truth, dtype=tf.int64)

                val_loss, val_accuracy, _ = compute_loss_and_accuracy(
                    model, val_fingerprints, val_ground_truth, training=False
                )
                conf_matrix = compute_confusion_matrix(
                    model, val_fingerprints, val_ground_truth, label_count
                )

                batch_size = min(FLAGS.batch_size, set_size - offset)
                total_accuracy += float(val_accuracy) * batch_size / set_size
                total_conf_matrix = (
                    conf_matrix
                    if total_conf_matrix is None
                    else total_conf_matrix + conf_matrix
                )

            with validation_writer.as_default():
                tf.summary.scalar("cross_entropy", val_loss, step=training_step)
                tf.summary.scalar("accuracy", total_accuracy, step=training_step)

            logger.info("Confusion Matrix:\n%s", total_conf_matrix.numpy())
            logger.info(
                "Step %d: Validation accuracy = %.1f%% (N=%d)",
                training_step, total_accuracy * 100, set_size,
            )

        # ---- Periodic checkpoint ----
        if training_step % FLAGS.save_step_interval == 0 or is_last_step:
            saved = ckpt_manager.save(checkpoint_number=training_step)
            logger.info("Saved checkpoint: %s", saved)

    # ---- Final test evaluation ----
    set_size = audio_processor.set_size("testing")
    logger.info("set_size=%d", set_size)
    total_accuracy = 0.0
    total_conf_matrix = None

    for offset in range(0, set_size, FLAGS.batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, offset, model_settings, 0.0, 0.0, 0, "testing", sess
        )
        test_fingerprints = tf.constant(test_fingerprints, dtype=tf.float32)
        test_ground_truth = tf.constant(test_ground_truth, dtype=tf.int64)

        _, test_accuracy, _ = compute_loss_and_accuracy(
            model, test_fingerprints, test_ground_truth, training=False
        )
        conf_matrix = compute_confusion_matrix(
            model, test_fingerprints, test_ground_truth, label_count
        )

        batch_size = min(FLAGS.batch_size, set_size - offset)
        total_accuracy += float(test_accuracy) * batch_size / set_size
        total_conf_matrix = (
            conf_matrix
            if total_conf_matrix is None
            else total_conf_matrix + conf_matrix
        )

    logger.warning("Confusion Matrix:\n%s", total_conf_matrix.numpy())
    logger.warning(
        "Final test accuracy = %.1f%% (N=%d)", total_accuracy * 100, set_size
    )

    # ---- Save final SavedModel ----
    saved_model_path = os.path.join(FLAGS.train_dir, "saved_model")
    model.save(saved_model_path)
    logger.info("SavedModel written to %s", saved_model_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_url",
        type=str,
        default="https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        help="Location of speech training data archive on the web.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tmp/speech_dataset/",
        help="Where to download the speech training data to.",
    )
    parser.add_argument(
        "--background_volume",
        type=float,
        default=0.1,
        help="How loud the background noise should be, between 0 and 1.",
    )
    parser.add_argument(
        "--background_frequency",
        type=float,
        default=0.8,
        help="How many of the training samples have background noise mixed in.",
    )
    parser.add_argument(
        "--silence_percentage",
        type=float,
        default=10.0,
        help="How much of the training data should be silence.",
    )
    parser.add_argument(
        "--unknown_percentage",
        type=float,
        default=10.0,
        help="How much of the training data should be unknown words.",
    )
    parser.add_argument(
        "--time_shift_ms",
        type=float,
        default=100.0,
        help="Range to randomly shift the training audio by in time.",
    )
    parser.add_argument(
        "--testing_percentage",
        type=int,
        default=10,
        help="What percentage of wavs to use as a test set.",
    )
    parser.add_argument(
        "--validation_percentage",
        type=int,
        default=10,
        help="What percentage of wavs to use as a validation set.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Expected sample rate of the wavs.",
    )
    parser.add_argument(
        "--clip_duration_ms",
        type=int,
        default=1000,
        help="Expected duration in milliseconds of the wavs.",
    )
    parser.add_argument(
        "--window_size_ms",
        type=float,
        default=30.0,
        help="How long each spectrogram timeslice is.",
    )
    parser.add_argument(
        "--window_stride_ms",
        type=float,
        default=10.0,
        help="How far to move in time between spectrogram timeslices.",
    )
    parser.add_argument(
        "--feature_bin_count",
        type=int,
        default=40,
        help="How many bins to use for the MFCC fingerprint.",
    )
    parser.add_argument(
        "--how_many_training_steps",
        type=str,
        default="15000,3000",
        help="How many training loops to run (comma-separated per phase).",
    )
    parser.add_argument(
        "--eval_step_interval",
        type=int,
        default=400,
        help="How often to evaluate the training results.",
    )
    parser.add_argument(
        "--learning_rate",
        type=str,
        default="0.001,0.0001",
        help="Learning rate per phase (comma-separated, matches --how_many_training_steps).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="How many items to train with at once.",
    )
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default="/tmp/retrain_logs",
        help="Where to save summary logs for TensorBoard.",
    )
    parser.add_argument(
        "--wanted_words",
        type=str,
        default="yes,no,up,down,left,right,on,off,stop,go",
        help="Words to use (others will be added to an unknown label).",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/tmp/speech_commands_train",
        help="Directory to write event logs and checkpoints.",
    )
    parser.add_argument(
        "--save_step_interval",
        type=int,
        default=100,
        help="Save model checkpoint every save_steps.",
    )
    parser.add_argument(
        "--start_checkpoint",
        type=str,
        default="",
        help="If specified, restore this pretrained model before any training.",
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="conv",
        help="What model architecture to use.",
    )
    parser.add_argument(
        "--check_nans",
        action="store_true",
        default=False,
        help="Whether to check for NaNs in gradients during training.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Whether to train the model for eight-bit deployment (requires TFLite post-training quantization).",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="mfcc",
        help='Spectrogram processing mode. Can be "mfcc", "average", or "micro".',
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"],
        help="Log verbosity.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="gradient_descent",
        choices=["gradient_descent", "momentum"],
        help="Optimizer to use.",
    )

    FLAGS = parser.parse_args()

    # Apply verbosity
    level = getattr(logging, FLAGS.verbosity.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

    main()