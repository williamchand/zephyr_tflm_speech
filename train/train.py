# TensorFlow Speech Commands Training (GPU-optimized, graph pipeline)

import argparse
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops

import input_data
import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FLAGS = None


# ------------------------------------------------------------
# GPU setup
# ------------------------------------------------------------

def configure_gpu(use_mixed_precision, use_xla):

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logger.info("GPUs detected: %s", gpus)

        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled")

        if use_xla:
            tf.config.optimizer.set_jit(True)
            logger.info("XLA enabled")

        strategy = tf.distribute.MirroredStrategy()

        logger.info(
            "MirroredStrategy replicas: %d",
            strategy.num_replicas_in_sync
        )

    else:
        logger.warning("No GPU found — using CPU")
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")

    return strategy


# ------------------------------------------------------------
# FAST graph tf.data pipeline
# ------------------------------------------------------------

def build_dataset(audio_processor,
                  model_settings,
                  mode,
                  background_frequency,
                  background_volume_range,
                  time_shift_samples,
                  batch_size,
                  shuffle):

    candidates = audio_processor.data_index[mode]

    files = [c["file"] for c in candidates]
    labels = [audio_processor.word_to_index[c["label"]] for c in candidates]

    desired_samples = model_settings["desired_samples"]
    window_size = model_settings["window_size_samples"]
    window_stride = model_settings["window_stride_samples"]
    fp_width = model_settings["fingerprint_width"]
    fp_size = model_settings["fingerprint_size"]
    sample_rate = model_settings["sample_rate"]

    AUTOTUNE = tf.data.AUTOTUNE

    def process_one(path, label):

        raw = tf.io.read_file(path)

        audio, _ = tf.audio.decode_wav(
            raw,
            desired_channels=1,
            desired_samples=desired_samples
        )

        # -------- time shift augmentation --------

        if time_shift_samples > 0:
            shift = tf.random.uniform(
                [],
                -time_shift_samples,
                time_shift_samples,
                dtype=tf.int32
            )

            audio = tf.roll(audio, shift, axis=0)

        # -------- spectrogram --------

        spectrogram = audio_ops.audio_spectrogram(
            audio,
            window_size=window_size,
            stride=window_stride,
            magnitude_squared=True
        )

        # -------- MFCC --------

        mfcc = audio_ops.mfcc(
            spectrogram,
            sample_rate,
            dct_coefficient_count=fp_width
        )

        mfcc = tf.reshape(mfcc, [-1])

        mfcc = mfcc[:fp_size]

        mfcc = tf.ensure_shape(mfcc, [fp_size])

        return mfcc, tf.cast(label, tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((files, labels))

    if shuffle:
        ds = ds.shuffle(len(files))

    ds = ds.map(
        process_one,
        num_parallel_calls=AUTOTUNE,
        deterministic=not shuffle
    )

    # optional: huge speedup after first epoch
    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)

    ds = ds.prefetch(AUTOTUNE)

    options = tf.data.Options()

    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True

    ds = ds.with_options(options)

    return ds


# ------------------------------------------------------------
# TRAIN STEP
# ------------------------------------------------------------

@tf.function
def train_step(model,
               optimizer,
               fingerprints,
               labels,
               global_batch_size):

    with tf.GradientTape() as tape:

        logits = model(fingerprints, training=True)

        logits = tf.cast(logits, tf.float32)

        per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels,
            logits,
            from_logits=True
        )

        loss = tf.nn.compute_average_loss(
            per_example_loss,
            global_batch_size=global_batch_size
        )

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(
        zip(grads, model.trainable_variables)
    )

    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(predictions, tf.cast(labels, tf.int64)),
            tf.float32
        )
    )

    return loss, accuracy


# ------------------------------------------------------------
# EVAL STEP
# ------------------------------------------------------------

@tf.function
def eval_step(model, fingerprints, labels, num_classes):

    logits = model(fingerprints, training=False)

    logits = tf.cast(logits, tf.float32)

    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels,
            logits,
            from_logits=True
        )
    )

    preds = tf.argmax(logits, axis=1, output_type=tf.int64)

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(preds, tf.cast(labels, tf.int64)),
            tf.float32
        )
    )

    conf = tf.math.confusion_matrix(
        tf.cast(labels, tf.int64),
        preds,
        num_classes=num_classes
    )

    return loss, accuracy, conf


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    gpus = tf.config.list_physical_devices("GPU")

    use_mp = (
        FLAGS.use_mixed_precision
        if FLAGS.use_mixed_precision is not None
        else bool(gpus)
    )

    strategy = configure_gpu(use_mp, FLAGS.use_xla)

    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(
            FLAGS.wanted_words.split(",")
        )),
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.feature_bin_count,
        FLAGS.preprocess
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
        FLAGS.summaries_dir
    )

    fingerprint_size = model_settings["fingerprint_size"]
    label_count = model_settings["label_count"]

    with strategy.scope():

        inputs = tf.keras.Input(shape=(fingerprint_size,))

        result = models.create_model(
            inputs,
            model_settings,
            FLAGS.model_architecture,
            is_training=True
        )

        outputs = result[0] if isinstance(result, tuple) else result

        if use_mp:
            outputs = tf.keras.layers.Activation(
                "linear",
                dtype="float32"
            )(outputs)

        model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    train_ds = build_dataset(
        audio_processor,
        model_settings,
        "training",
        FLAGS.background_frequency,
        FLAGS.background_volume,
        0,
        FLAGS.batch_size,
        shuffle=True
    )

    train_ds = train_ds.repeat()

    train_iter = iter(
        strategy.experimental_distribute_dataset(train_ds)
    )

    for step in range(1, 10001):

        fingerprints, labels = next(train_iter)

        loss_r, acc_r = strategy.run(
            train_step,
            args=(
                model,
                optimizer,
                fingerprints,
                labels,
                FLAGS.batch_size
            )
        )

        loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            loss_r,
            axis=None
        )

        acc = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            acc_r,
            axis=None
        )

        if step % 100 == 0:

            logger.info(
                "step %d  loss %.4f  acc %.2f%%",
                step,
                float(loss),
                float(acc) * 100
            )


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_url", type=str)
    parser.add_argument("--data_dir", type=str)

    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument(
        "--wanted_words",
        type=str,
        default="yes,no,up,down,left,right,on,off,stop,go"
    )

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--clip_duration_ms", type=int, default=1000)

    parser.add_argument("--window_size_ms", type=float, default=30)
    parser.add_argument("--window_stride_ms", type=float, default=10)

    parser.add_argument("--feature_bin_count", type=int, default=40)

    parser.add_argument("--preprocess", type=str, default="mfcc")
    parser.add_argument("--model_architecture", type=str, default="conv")

    parser.add_argument("--background_frequency", type=float, default=0.8)
    parser.add_argument("--background_volume", type=float, default=0.1)

    parser.add_argument("--silence_percentage", type=float, default=10)
    parser.add_argument("--unknown_percentage", type=float, default=10)

    parser.add_argument("--validation_percentage", type=int, default=10)
    parser.add_argument("--testing_percentage", type=int, default=10)

    parser.add_argument("--summaries_dir", type=str, default="/tmp/logs")

    parser.add_argument(
        "--use_mixed_precision",
        type=lambda x: x.lower() != "false",
        nargs="?",
        const=True,
        default=None
    )

    parser.add_argument("--use_xla", action="store_true")

    FLAGS = parser.parse_args()

    main()