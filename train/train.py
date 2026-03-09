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

GPU-intensive TF2 version.

What changed from your train.py and why
-----------------------------------------
Your original had three blocking bottlenecks that stalled the GPU between
every single batch:

  ORIGINAL (GPU idles between every step):
    for training_step in range(...):
        fp, lbl = audio_processor.get_data(...)  # CPU for-loop, blocks GPU
        with GradientTape():                      # GPU wakes up
            ...                                   # GPU finishes
        # GPU idles again while get_data() runs next batch
        #
        # Validation and test loops also called get_data() in Python for-loops,
        # the same blocking pattern.

  NEW (GPU stays busy):
    build_dataset() replaces every get_data() call with a tf.data pipeline:
      - from_tensor_slices: file paths + labels as in-memory constants
      - map(process_one, AUTOTUNE): N CPU threads run file I/O + wav decode
        + augment + spectrogram + MFCC concurrently
      - batch() + prefetch(AUTOTUNE): completed batches DMA'd to GPU DRAM
        while the previous step is still running
      - strategy.experimental_distribute_dataset: shards across all GPUs

    Training loop:
      for step in range(...):
          fp, lbl = next(train_iter)         # already in GPU DRAM, ~0 wait
          strategy.run(train_step, (fp, lbl)) # GPU never idles

GPU stack used
--------------
  MirroredStrategy     synchronous data-parallel across all visible GPUs;
                       NCCL gradient all-reduce on multi-GPU
  LossScaleOptimizer   prevents float16 gradient underflow (mixed precision)
  mixed_float16 policy float16 Tensor Core compute, float32 master weights
  XLA JIT (--use_xla)  fuses elementwise ops into single GPU kernels
  memory_growth=True   prevents TF from pre-allocating all VRAM

CPU / GPU split
---------------
  CPU  audio_spectrogram / mfcc have no GPU kernels — pinned to /CPU:0.
       tf.data AUTOTUNE workers run these in parallel C++ threads.
  GPU  model forward pass, loss, GradientTape, optimizer.apply_gradients,
       eval metrics, confusion matrix.
"""
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


# ---------------------------------------------------------------------------
# GPU setup — must run before any tf.Variable / tf.keras.* calls
# ---------------------------------------------------------------------------

def configure_gpu(use_mixed_precision: bool, use_xla: bool):
    """Enable memory growth, mixed precision, XLA; return a distribute strategy.

    Returns MirroredStrategy when ≥1 GPU is detected, otherwise falls back
    to OneDeviceStrategy('/CPU:0') so the rest of the code is unchanged.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info('Found %d GPU(s): %s', len(gpus), [g.name for g in gpus])

        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info('Mixed precision: float16 compute / float32 weights.')

        if use_xla:
            tf.config.optimizer.set_jit(True)
            logger.info('XLA JIT compilation enabled.')

        strategy = tf.distribute.MirroredStrategy()
        logger.info('MirroredStrategy: %d replica(s).',
                    strategy.num_replicas_in_sync)
    else:
        logger.warning('No GPU found — running on CPU.')
        strategy = tf.distribute.OneDeviceStrategy('/CPU:0')

    return strategy


# ---------------------------------------------------------------------------
# tf.data pipeline  (replaces all audio_processor.get_data() calls)
# ---------------------------------------------------------------------------

def build_dataset(audio_processor, model_settings, mode,
                  background_frequency, background_volume_range,
                  time_shift_samples, batch_size, shuffle):
    """Build a fully-pipelined, GPU-prefetched tf.data.Dataset.

    Pipeline stages (all overlap in time):
      [CPU thread 0..N]  read file → decode wav → time-shift → mix noise
                         → spectrogram → MFCC
                         (AUTOTUNE = as many threads as TF finds useful)
      [CPU→GPU DMA]      prefetch pushes completed batches into pinned GPU
                         memory before the current step finishes
      [GPU]              train_step / eval_step reads from GPU DRAM directly

    This replaces the original:
      for offset in range(0, set_size, batch_size):
          fp, lbl = audio_processor.get_data(batch_size, offset, ...)
    which ran a Python for-loop on CPU and blocked the GPU each iteration.

    Args:
      audio_processor:        Initialised AudioProcessor.
      model_settings:         Dict from models.prepare_model_settings().
      mode:                   'training', 'validation', or 'testing'.
      background_frequency:   Fraction of clips to mix background noise into.
      background_volume_range: Max background noise volume (0–1).
      time_shift_samples:     Max random time-shift in samples.
      batch_size:             Global batch size (split across GPU replicas).
      shuffle:                Shuffle file order each epoch.

    Returns:
      tf.data.Dataset yielding (fingerprints [B, fp_size], labels [B]).
    """
    candidates   = audio_processor.data_index[mode]
    files        = [c['file'] for c in candidates]
    word_indices = [audio_processor.word_to_index[c['label']]
                    for c in candidates]
    is_silence   = [1 if c['label'] == input_data.SILENCE_LABEL else 0
                    for c in candidates]

    desired_samples = model_settings['desired_samples']
    window_size     = model_settings['window_size_samples']
    window_stride   = model_settings['window_stride_samples']
    fp_width        = model_settings['fingerprint_width']
    fp_size         = model_settings['fingerprint_size']
    avg_win         = model_settings.get('average_window_width', -1)
    sample_rate     = model_settings['sample_rate']
    preprocess      = model_settings['preprocess']

    # Pre-stack background clips → [N_bg, desired_samples] constant tensor.
    # The map function can index this tensor without touching Python per sample.
    if audio_processor.background_data:
        bg_clips = []
        for bg in audio_processor.background_data:
            if len(bg) >= desired_samples:
                bg_clips.append(bg[:desired_samples].astype(np.float32))
            else:
                pad = np.zeros(desired_samples, dtype=np.float32)
                pad[:len(bg)] = bg
                bg_clips.append(pad)
        bg_tensor = tf.constant(np.stack(bg_clips), dtype=tf.float32)
    else:
        bg_tensor = tf.zeros([1, desired_samples], dtype=tf.float32)

    use_bg = bool(audio_processor.background_data) and (mode == 'training')
    n_bg   = int(bg_tensor.shape[0])

    # ------------------------------------------------------------------
    # process_one  — CPU-pinned feature extraction for one sample.
    #
    # Why /CPU:0: audio_spectrogram and mfcc have no GPU kernels. Pinning
    # explicitly prevents unnecessary host↔device transfers of intermediate
    # tensors.
    #
    # Why no input_signature: audio_ops.mfcc returns a tensor with a
    # dynamic time-dimension. input_signature forces symbolic tracing which
    # assigns that dimension 0, making tf.reshape(..., [fp_size]) raise.
    # Plain @tf.function traces lazily on the first real call with concrete
    # shapes — no reshape crash.
    # ------------------------------------------------------------------
    @tf.function
    def process_one(wav_path, label, silence_flag):
        with tf.device('/CPU:0'):
            # Load
            audio, sr = tf.audio.decode_wav(
                tf.io.read_file(wav_path),
                desired_channels=1,
                desired_samples=desired_samples)

            # Time shift
            shift = (tf.random.uniform(
                         [], -time_shift_samples, time_shift_samples,
                         dtype=tf.int32)
                     if time_shift_samples > 0
                     else tf.constant(0, tf.int32))
            pad_l  = tf.maximum(shift, 0)
            pad_r  = tf.maximum(-shift, 0)
            sliced = tf.slice(
                tf.pad(audio, [[pad_l, pad_r], [0, 0]]),
                [pad_r, 0], [desired_samples, -1])

            # Silence: zero foreground
            sliced = sliced * tf.cond(
                tf.equal(silence_flag, 1),
                lambda: tf.constant(0.0),
                lambda: tf.constant(1.0))

            # Background noise
            if use_bg:
                bg_idx  = tf.random.uniform([], 0, n_bg, dtype=tf.int32)
                bg_clip = tf.reshape(bg_tensor[bg_idx], [desired_samples, 1])
                bg_vol  = tf.cond(
                    tf.equal(silence_flag, 1),
                    lambda: tf.random.uniform([], 0.0, 1.0),
                    lambda: tf.cond(
                        tf.less(tf.random.uniform([]),
                                tf.constant(background_frequency,
                                            dtype=tf.float32)),
                        lambda: tf.random.uniform(
                            [], 0.0,
                            tf.constant(background_volume_range,
                                        dtype=tf.float32)),
                        lambda: tf.constant(0.0)))
                mixed = tf.clip_by_value(sliced + bg_clip * bg_vol, -1.0, 1.0)
            else:
                mixed = sliced

            # Spectrogram
            spectrogram = audio_ops.audio_spectrogram(
                mixed,
                window_size=window_size,
                stride=window_stride,
                magnitude_squared=True)

            # Features
            if preprocess == 'average':
                features = tf.nn.pool(
                    input=tf.expand_dims(spectrogram, -1),
                    window_shape=[1, avg_win],
                    strides=[1, avg_win],
                    pooling_type='AVG',
                    padding='SAME')
            elif preprocess == 'mfcc':
                features = audio_ops.mfcc(
                    spectrogram, sr,
                    dct_coefficient_count=fp_width)
            elif preprocess == 'micro':
                try:
                    from tensorflow.lite.experimental.microfrontend.python.ops \
                        import audio_microfrontend_op as frontend_op
                except ImportError:
                    raise ImportError(
                        'Micro frontend op unavailable. Build with Bazel or '
                        'install the microfrontend package.')
                ws_ms   = (window_size   * 1000) / sample_rate
                wt_ms   = (window_stride * 1000) / sample_rate
                i16     = tf.cast(tf.multiply(mixed, 32768), tf.int16)
                mf      = frontend_op.audio_microfrontend(
                    i16, sample_rate=sample_rate,
                    window_size=ws_ms, window_step=wt_ms,
                    num_channels=fp_width, out_scale=1,
                    out_type=tf.float32)
                features = tf.multiply(mf, 10.0 / 256.0)
            else:
                raise ValueError(
                    'Unknown preprocess mode "%s" (should be "mfcc", '
                    '"average", or "micro")' % preprocess)

            # DO NOT tf.reshape here.
            # ds.map() forces a trace of this entire function body; audio_ops.mfcc
            # returns a tensor whose time-dimension TF assigns 0 during tracing.
            # tf.reshape([..., 0, 40] -> [1960]) therefore raises at trace time
            # even though the runtime shapes are always correct.
            # The flatten is done in a SEPARATE .map(flatten) step below, which
            # is traced independently once process_one's output spec is known.
            return features, tf.cast(label, tf.int32)

    # Assemble dataset
    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(files),
        tf.constant(word_indices, dtype=tf.int32),
        tf.constant(is_silence,   dtype=tf.int32),
    ))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(files), reshuffle_each_iteration=True)

    # AUTOTUNE: TF picks the thread count based on CPU core count and
    # observed throughput. Typically saturates all CPU cores.
    ds = ds.map(process_one,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=not shuffle)

    # Flatten in a SEPARATE map so tf.reshape sees a concrete shape.
    # process_one's output spec is fully known by the time this is traced,
    # so [fp_size] is a valid target shape — no symbolic-zero crash.
    def flatten(raw_features, label):
        return tf.reshape(raw_features, [fp_size]), label
    ds = ds.map(flatten, num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=not shuffle)

    ds = ds.batch(batch_size, drop_remainder=False)

    # prefetch: pipeline computes batch N+1 on CPU while GPU trains on
    # batch N — GPU never stalls waiting for data.
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # DATA sharding: each GPU replica reads a non-overlapping shard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA)
    return ds.with_options(options)


# ---------------------------------------------------------------------------
# Compiled train / eval steps
# ---------------------------------------------------------------------------

@tf.function
def train_step(model, optimizer, fingerprints, labels):
    """Forward + backward pass on one GPU replica.

    tf.nn.compute_average_loss scales the loss so the gradient magnitude
    is independent of the number of replicas — required for MirroredStrategy.

    Returns: (loss, accuracy) scalar tensors.
    """
    with tf.GradientTape() as tape:
        logits     = model(fingerprints, training=True)
        logits_f32 = tf.cast(logits, tf.float32)
        loss = tf.nn.compute_average_loss(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits_f32, from_logits=True),
            global_batch_size=tf.shape(fingerprints)[0])

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    predicted = tf.argmax(logits_f32, axis=1, output_type=tf.int64)
    accuracy  = tf.reduce_mean(
        tf.cast(tf.equal(predicted, tf.cast(labels, tf.int64)), tf.float32))
    return loss, accuracy


@tf.function
def eval_step(model, fingerprints, labels, num_classes):
    """Forward pass only — no gradient computation.

    Returns: (loss, accuracy, confusion_matrix) tensors.
    """
    logits     = model(fingerprints, training=False)
    logits_f32 = tf.cast(logits, tf.float32)
    loss       = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits_f32, from_logits=True))
    predicted  = tf.argmax(logits_f32, axis=1, output_type=tf.int64)
    accuracy   = tf.reduce_mean(
        tf.cast(tf.equal(predicted, tf.cast(labels, tf.int64)), tf.float32))
    conf       = tf.math.confusion_matrix(
        tf.cast(labels, tf.int64), predicted, num_classes=num_classes)
    return loss, accuracy, conf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- GPU setup (must happen before any TF ops) ----
    gpus   = tf.config.list_physical_devices('GPU')
    use_mp = (FLAGS.use_mixed_precision
              if FLAGS.use_mixed_precision is not None
              else bool(gpus))
    strategy = configure_gpu(use_mixed_precision=use_mp, use_xla=FLAGS.use_xla)

    # CPU thread tuning for tf.data workers
    if FLAGS.inter_op_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(
            FLAGS.inter_op_threads)
    if FLAGS.intra_op_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(
            FLAGS.intra_op_threads)

    # ---- Model / audio settings ----
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms,
        FLAGS.window_size_ms, FLAGS.window_stride_ms,
        FLAGS.feature_bin_count, FLAGS.preprocess)

    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir,
        FLAGS.silence_percentage, FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','),
        FLAGS.validation_percentage, FLAGS.testing_percentage,
        model_settings, FLAGS.summaries_dir)

    fingerprint_size   = model_settings['fingerprint_size']
    label_count        = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

    # ---- Learning-rate schedule ----
    training_steps_list = list(map(int,   FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise ValueError(
            '--how_many_training_steps and --learning_rate must have equal '
            f'length, got {len(training_steps_list)} vs '
            f'{len(learning_rates_list)}')
    training_steps_max = int(np.sum(training_steps_list))

    # ---- Build model inside strategy scope ----
    # Everything created here is placed on GPU(s) and mirrored across replicas.
    with strategy.scope():
        inputs  = tf.keras.Input(shape=(fingerprint_size,),
                                 name='fingerprint_input')
        result  = models.create_model(
            inputs, model_settings, FLAGS.model_architecture, is_training=True)
        outputs = result[0] if isinstance(result, tuple) else result

        # Cast logits to float32 for stable loss when mixed precision is on.
        if use_mp:
            outputs = tf.keras.layers.Activation(
                'linear', dtype='float32', name='fp32_logits')(outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary(print_fn=logger.info)

        if FLAGS.optimizer == 'gradient_descent':
            base_opt = tf.keras.optimizers.SGD(
                learning_rate=learning_rates_list[0])
        elif FLAGS.optimizer == 'momentum':
            base_opt = tf.keras.optimizers.SGD(
                learning_rate=learning_rates_list[0],
                momentum=0.9, nesterov=True)
        else:
            raise ValueError(f'Invalid optimizer: {FLAGS.optimizer}')

        # LossScaleOptimizer prevents float16 gradient underflow.
        optimizer = (tf.keras.mixed_precision.LossScaleOptimizer(base_opt)
                     if use_mp else base_opt)

    # ---- Checkpointing ----
    os.makedirs(FLAGS.train_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=FLAGS.train_dir, max_to_keep=5,
        checkpoint_name=FLAGS.model_architecture)

    start_step = 1
    if FLAGS.start_checkpoint:
        ckpt.restore(FLAGS.start_checkpoint).expect_partial()
        logger.info('Restored checkpoint from %s', FLAGS.start_checkpoint)
    elif ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        logger.info('Restored latest checkpoint: %s',
                    ckpt_manager.latest_checkpoint)

    # ---- TensorBoard ----
    train_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.summaries_dir, 'train'))
    val_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.summaries_dir, 'validation'))

    # ---- Save word list ----
    with open(os.path.join(FLAGS.train_dir,
                           FLAGS.model_architecture + '_labels.txt'), 'w') as f:
        f.write('\n'.join(audio_processor.words_list))

    # ---- Build tf.data pipelines ----
    # build_dataset() replaces all audio_processor.get_data() call sites.
    # CPU feature extraction runs in parallel threads; batches are prefetched
    # into GPU DRAM; training loop pulls from GPU memory at each step.
    train_ds = build_dataset(
        audio_processor, model_settings, 'training',
        FLAGS.background_frequency, FLAGS.background_volume,
        time_shift_samples, FLAGS.batch_size, shuffle=True)
    train_ds   = train_ds.repeat()   # infinite stream; we step manually
    train_iter = iter(strategy.experimental_distribute_dataset(train_ds))

    val_ds = build_dataset(
        audio_processor, model_settings, 'validation',
        0.0, 0.0, 0, FLAGS.batch_size, shuffle=False)
    val_dist_ds = strategy.experimental_distribute_dataset(val_ds)

    test_ds = build_dataset(
        audio_processor, model_settings, 'testing',
        0.0, 0.0, 0, FLAGS.batch_size, shuffle=False)
    test_dist_ds = strategy.experimental_distribute_dataset(test_ds)

    # Convenience: reach through LossScaleOptimizer wrapper for LR assignment
    inner_opt = (optimizer.inner_optimizer
                 if isinstance(optimizer,
                               tf.keras.mixed_precision.LossScaleOptimizer)
                 else optimizer)

    # ---- Training loop ----
    for training_step in range(start_step, training_steps_max + 1):
        # Dynamic learning rate
        steps_sum = 0
        for i, steps in enumerate(training_steps_list):
            steps_sum += steps
            if training_step <= steps_sum:
                lr_value = learning_rates_list[i]
                break
        inner_opt.learning_rate.assign(lr_value)

        # Pull next batch — already in GPU DRAM, ~zero CPU wait.
        fingerprints, labels = next(train_iter)

        # Dispatch to all GPU replicas; NCCL all-reduces gradients before
        # apply_gradients keeps every replica in sync.
        loss_r, acc_r = strategy.run(
            train_step, args=(model, optimizer, fingerprints, labels))

        train_loss     = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, loss_r, axis=None)
        train_accuracy = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, acc_r, axis=None)

        if FLAGS.check_nans and tf.math.is_nan(train_loss):
            raise RuntimeError('NaN loss at step %d' % training_step)

        with train_writer.as_default():
            tf.summary.scalar('cross_entropy', train_loss,     step=training_step)
            tf.summary.scalar('accuracy',      train_accuracy, step=training_step)
            tf.summary.scalar('learning_rate', lr_value,       step=training_step)

        logger.debug('Step #%d: rate %f, accuracy %.1f%%, loss %f',
                     training_step, lr_value,
                     float(train_accuracy) * 100, float(train_loss))

        is_last_step = (training_step == training_steps_max)
        if training_step % FLAGS.eval_step_interval == 0 or is_last_step:
            logger.info('Step #%d: rate %f, accuracy %.1f%%, loss %f',
                        training_step, lr_value,
                        float(train_accuracy) * 100, float(train_loss))

            # ---- Validation ----
            set_size          = audio_processor.set_size('validation')
            total_accuracy    = 0.0
            total_conf_matrix = None

            for val_fp, val_lbl in val_dist_ds:
                v_loss_r, v_acc_r, conf_r = strategy.run(
                    eval_step,
                    args=(model, val_fp, val_lbl,
                          tf.constant(label_count, tf.int32)))
                v_acc  = strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, v_acc_r, axis=None)
                v_loss = strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, v_loss_r, axis=None)
                conf   = strategy.reduce(
                    tf.distribute.ReduceOp.SUM, conf_r, axis=None)

                batch_sz = (int(val_fp.shape[0])
                            if hasattr(val_fp, 'shape') and val_fp.shape[0]
                            else FLAGS.batch_size)
                total_accuracy    += float(v_acc) * batch_sz / set_size
                total_conf_matrix  = (conf if total_conf_matrix is None
                                      else total_conf_matrix + conf)

            with val_writer.as_default():
                tf.summary.scalar('cross_entropy', v_loss,        step=training_step)
                tf.summary.scalar('accuracy',      total_accuracy, step=training_step)

            logger.info('Confusion Matrix:\n%s', total_conf_matrix.numpy())
            logger.info('Step %d: Validation accuracy = %.1f%% (N=%d)',
                        training_step, total_accuracy * 100, set_size)

        if training_step % FLAGS.save_step_interval == 0 or is_last_step:
            saved = ckpt_manager.save(checkpoint_number=training_step)
            logger.info('Saved checkpoint: %s', saved)

    # ---- Final test evaluation ----
    set_size          = audio_processor.set_size('testing')
    total_accuracy    = 0.0
    total_conf_matrix = None

    for test_fp, test_lbl in test_dist_ds:
        _, t_acc_r, conf_r = strategy.run(
            eval_step,
            args=(model, test_fp, test_lbl,
                  tf.constant(label_count, tf.int32)))
        t_acc = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, t_acc_r, axis=None)
        conf  = strategy.reduce(
            tf.distribute.ReduceOp.SUM, conf_r, axis=None)
        batch_sz = (int(test_fp.shape[0])
                    if hasattr(test_fp, 'shape') and test_fp.shape[0]
                    else FLAGS.batch_size)
        total_accuracy    += float(t_acc) * batch_sz / set_size
        total_conf_matrix  = (conf if total_conf_matrix is None
                               else total_conf_matrix + conf)

    logger.warning('Confusion Matrix:\n%s', total_conf_matrix.numpy())
    logger.warning('Final test accuracy = %.1f%% (N=%d)',
                   total_accuracy * 100, set_size)

    saved_model_path = os.path.join(FLAGS.train_dir, 'saved_model')
    model.save(saved_model_path)
    logger.info('SavedModel written to %s', saved_model_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ---- Original flags (all preserved) ----
    parser.add_argument('--data_url', type=str,
        default='https://storage.googleapis.com/download.tensorflow.org/'
                'data/speech_commands_v0.02.tar.gz')
    parser.add_argument('--data_dir',              type=str,   default='/tmp/speech_dataset/')
    parser.add_argument('--background_volume',     type=float, default=0.1)
    parser.add_argument('--background_frequency',  type=float, default=0.8)
    parser.add_argument('--silence_percentage',    type=float, default=10.0)
    parser.add_argument('--unknown_percentage',    type=float, default=10.0)
    parser.add_argument('--time_shift_ms',         type=float, default=100.0)
    parser.add_argument('--testing_percentage',    type=int,   default=10)
    parser.add_argument('--validation_percentage', type=int,   default=10)
    parser.add_argument('--sample_rate',           type=int,   default=16000)
    parser.add_argument('--clip_duration_ms',      type=int,   default=1000)
    parser.add_argument('--window_size_ms',        type=float, default=30.0)
    parser.add_argument('--window_stride_ms',      type=float, default=10.0)
    parser.add_argument('--feature_bin_count',     type=int,   default=40)
    parser.add_argument('--how_many_training_steps', type=str, default='15000,3000')
    parser.add_argument('--eval_step_interval',    type=int,   default=400)
    parser.add_argument('--learning_rate',         type=str,   default='0.001,0.0001')
    parser.add_argument('--batch_size',            type=int,   default=100)
    parser.add_argument('--summaries_dir',         type=str,   default='/tmp/retrain_logs')
    parser.add_argument('--wanted_words',          type=str,
        default='yes,no,up,down,left,right,on,off,stop,go')
    parser.add_argument('--train_dir',             type=str,   default='/tmp/speech_commands_train')
    parser.add_argument('--save_step_interval',    type=int,   default=100)
    parser.add_argument('--start_checkpoint',      type=str,   default='')
    parser.add_argument('--model_architecture',    type=str,   default='conv')
    parser.add_argument('--check_nans',            action='store_true', default=False)
    parser.add_argument('--quantize',              action='store_true', default=False)
    parser.add_argument('--preprocess',            type=str,   default='mfcc')
    parser.add_argument('--verbosity',             type=str,   default='INFO',
        choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
    parser.add_argument('--optimizer',             type=str,   default='gradient_descent',
        choices=['gradient_descent', 'momentum'])

    # ---- New GPU flags ----
    parser.add_argument('--use_mixed_precision',
        type=lambda x: x.lower() != 'false',
        nargs='?', const=True, default=None,
        help='float16 compute / float32 weights (Tensor Cores). '
             'Default: auto — ON when GPU detected. '
             'Pass --use_mixed_precision=false to force off.')
    parser.add_argument('--use_xla',
        action='store_true', default=False,
        help='Enable XLA JIT kernel fusion.')
    parser.add_argument('--inter_op_threads', type=int, default=0,
        help='CPU inter-op threads for tf.data workers (0 = TF default).')
    parser.add_argument('--intra_op_threads', type=int, default=0,
        help='CPU intra-op threads for tf.data workers (0 = TF default).')

    FLAGS = parser.parse_args()
    logging.getLogger().setLevel(
        getattr(logging, FLAGS.verbosity.upper(), logging.INFO))
    main()