# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""TF2 keyword-spotting trainer — GPU-accelerated, same preprocessing as train.py.

Architecture parity with models.py
────────────────────────────────────
All six architectures from models.py are implemented as native Keras models so
they can be trained with model.fit() and GPU acceleration:

  single_fc            – single Dense layer
  conv                 – two Conv2D blocks + Dense (cnn-trad-fpool3)
  low_latency_conv     – single full-height Conv2D + three Dense layers
  low_latency_svdf     – SVDFLayer (custom Keras layer) + three Dense layers
  tiny_conv            – one strided Conv2D + Dense  (<20 KB RAM target)
  tiny_embedding_conv  – two strided Conv2D + Dense  (<20 KB RAM target)

SVDF implementation notes
──────────────────────────
The original TF1 SVDF uses a non-trainable variable 'runtime-memory' that is
updated in-place during *inference* to slide a context window.  This state
makes it incompatible with stateless Keras training.

Best-practice solution (used here):
  • Training   — SVDFLayer processes the full fingerprint (all time frames) in
                 a single pass with no state, exactly as the original does for
                 is_training=True.  The frequency and time filter weights are
                 the same shape as in models.py.
  • Inference  — A companion SVDFStreamingModel wraps the trained weights and
                 maintains a rolling context buffer as a tf.Variable so that
                 it can be called one frame at a time, matching the original
                 runtime-memory behaviour.  This model is saved alongside the
                 batch SavedModel and is the one that should be used for
                 freeze / TFLite conversion.

Preprocessing
─────────────
Uses input_data.AudioProcessor.get_data() via a TF1 session (the only way to
produce mfcc / micro / average fingerprints identical to train.py).  The
numpy result is wrapped as a tf.data.Dataset for GPU training.

GPU acceleration
────────────────
  • MirroredStrategy  – single or multi-GPU
  • mixed_float16     – fp16 activations, fp32 master weights (GPU only)
  • jit_compile=True  – XLA fusion of forward + backward pass
  • tf.data pipeline  – cached, shuffled, prefetched to hide CPU latency

Example
───────
    python train_tf2.py \\
        --data_url=https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz \\
        --data_dir=dataset/ \\
        --wanted_words=yes,no,up,down,left,right,stop,go \\
        --silence_percentage=10 --unknown_percentage=10 \\
        --preprocess=micro --window_stride=20 \\
        --model_architecture=tiny_conv \\
        --background_frequency=0.8 --background_volume=0.1 \\
        --how_many_training_steps=20000,10000 \\
        --learning_rate=0.001,0.0001 \\
        --saved_model_dir=models/saved_model_tf2
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import input_data
import models as model_settings_lib


# ============================================================================
# GPU / accelerator setup
# ============================================================================

def configure_gpu():
    """Enables memory growth and returns (MirroredStrategy, gpu_count).

    Must be called before any TF variable or dataset is created.
    Mixed precision is enabled on GPU for ~2× throughput; reverts to float32
    on CPU where fp16 gives no benefit.
    """
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f'[GPU] {len(gpus)} GPU(s) detected. Using MirroredStrategy.')
        print(f'[GPU] Precision: '
              f'{tf.keras.mixed_precision.global_policy().name}')
    else:
        tf.keras.mixed_precision.set_global_policy('float32')
        print('[GPU] No GPU — training on CPU (float32).')

    return tf.distribute.MirroredStrategy(), len(gpus)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='TF2 keyword-spotting trainer '
                    '(same preprocessing as train.py)')

    # Dataset — identical flags to train.py
    p.add_argument('--data_url', type=str,
        default='https://storage.googleapis.com/download.tensorflow.org/'
                'data/speech_commands_v0.02.tar.gz')
    p.add_argument('--data_dir',            type=str,   default='dataset/')
    p.add_argument('--wanted_words',        type=str,
        default='yes,no,up,down,left,right,stop,go')
    p.add_argument('--silence_percentage',  type=float, default=10.0)
    p.add_argument('--unknown_percentage',  type=float, default=10.0)
    p.add_argument('--validation_percentage', type=float, default=10.0)
    p.add_argument('--testing_percentage',  type=float, default=10.0)

    # Audio / preprocessing — identical flags to train.py
    p.add_argument('--sample_rate',         type=int,   default=16000)
    p.add_argument('--clip_duration_ms',    type=int,   default=1000)
    p.add_argument('--window_size_ms',      type=float, default=30.0)
    p.add_argument('--window_stride',       type=float, default=20.0,
        help='Window stride in ms (window_stride_ms in freeze.py)')
    p.add_argument('--feature_bin_count',   type=int,   default=40)
    p.add_argument('--preprocess',          type=str,   default='micro',
        choices=['mfcc', 'average', 'micro'])
    p.add_argument('--background_frequency', type=float, default=0.8)
    p.add_argument('--background_volume',   type=float, default=0.1)
    p.add_argument('--time_shift_ms',       type=float, default=100.0)

    # Model
    p.add_argument('--model_architecture',  type=str,   default='tiny_conv',
        choices=['single_fc', 'conv', 'low_latency_conv',
                 'low_latency_svdf', 'tiny_conv', 'tiny_embedding_conv'])

    # Training
    p.add_argument('--how_many_training_steps', type=str,
        default='20000,10000')
    p.add_argument('--learning_rate',       type=str,   default='0.001,0.0001')
    p.add_argument('--batch_size',          type=int,   default=100,
        help='Per-replica batch size.')

    # Logging / output
    p.add_argument('--summaries_dir',       type=str,   default='logs/')
    p.add_argument('--train_dir',           type=str,   default='train/')
    p.add_argument('--saved_model_dir',     type=str,
        default='models/saved_model_tf2')
    p.add_argument('--plot_dir',            type=str,   default='models')

    return p.parse_args()


# ============================================================================
# tf.data pipeline
# ============================================================================

def make_dataset(audio_proc, model_settings, mode, batch_size,
                 background_frequency, background_volume, time_shift_samples,
                 num_gpus, shuffle):
    """Wraps AudioProcessor.get_data() as a GPU-ready tf.data.Dataset.

    AudioProcessor.get_data() is called directly (TF2 eager).
    TF1 audio_spectrogram / microfrontend C++ ops.  The resulting numpy
    arrays are handed to tf.data for batching, caching, shuffling and
    prefetching so the GPU pipeline is never starved.

    Args:
        audio_proc:            input_data.AudioProcessor instance.
        model_settings:        Dict from model_settings_lib.prepare_model_settings().
        mode:                  'training', 'validation', or 'testing'.
        batch_size:            Per-replica batch size.
        background_frequency:  Fraction of clips to augment with noise.
        background_volume:     Max noise volume.
        time_shift_samples:    Max time-shift (training only).
        num_gpus:              Replica count (scales total batch size).
        shuffle:               Whether to shuffle the dataset.

    Returns:
        Batched tf.data.Dataset of (fingerprints float32, labels int32).
    """
    fingerprints, labels = audio_proc.get_data(
        how_many=-1,
        offset=0,
        model_settings=model_settings,
        background_frequency=background_frequency,
        background_volume_range=background_volume,
        time_shift=time_shift_samples if mode == 'training' else 0,
        mode=mode,
    )

    global_batch = batch_size * max(num_gpus, 1)
    ds = tf.data.Dataset.from_tensor_slices(
        (fingerprints.astype(np.float32), labels.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(len(fingerprints), reshuffle_each_iteration=True)
    # drop_remainder=True for training keeps batch shapes static,
    # which is required for XLA jit_compile to compile one kernel per step.
    ds = (ds
          .batch(global_batch, drop_remainder=(mode == 'training'))
          .cache()
          .prefetch(tf.data.AUTOTUNE))
    return ds


# ============================================================================
# Keras model builders  (exact ports of all architectures in models.py)
# ============================================================================

def _truncated_normal(stddev):
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)


def _flat_input(model_settings):
    """Returns a Keras Input for the flat fingerprint vector."""
    return tf.keras.Input(
        shape=(model_settings['fingerprint_size'],),
        name='fingerprint_input')


# ----------------------------------------------------------------------------
# single_fc
# ----------------------------------------------------------------------------

def build_single_fc(model_settings):
    """Port of create_single_fc_model.

    fingerprint → Dense(label_count)
    """
    inp = _flat_input(model_settings)
    out = tf.keras.layers.Dense(
        model_settings['label_count'],
        kernel_initializer=_truncated_normal(0.001),
        bias_initializer='zeros',
        dtype='float32', name='logits')(inp)
    return tf.keras.Model(inp, out, name='single_fc')


# ----------------------------------------------------------------------------
# conv  (cnn-trad-fpool3)
# ----------------------------------------------------------------------------

def build_conv(model_settings):
    """Port of create_conv_model.

    fingerprint → reshape(time, freq, 1)
      → Conv2D(64, 20×8, SAME) → ReLU → Dropout(0.5) → MaxPool(2×2)
      → Conv2D(64, 10×4, SAME) → ReLU → Dropout(0.5)
      → Flatten → Dense(label_count)
    """
    fw = model_settings['fingerprint_width']
    ft = model_settings['spectrogram_length']
    inp = _flat_input(model_settings)
    x   = tf.keras.layers.Reshape((ft, fw, 1))(inp)

    x   = tf.keras.layers.Conv2D(64, (20, 8), padding='same',
              activation='relu',
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros')(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x   = tf.keras.layers.Conv2D(64, (10, 4), padding='same',
              activation='relu',
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros')(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(
              model_settings['label_count'],
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros',
              dtype='float32', name='logits')(x)
    return tf.keras.Model(inp, out, name='conv')


# ----------------------------------------------------------------------------
# low_latency_conv  (cnn-one-fstride4)
# ----------------------------------------------------------------------------

def build_low_latency_conv(model_settings):
    """Port of create_low_latency_conv_model.

    fingerprint → reshape(time, freq, 1)
      → Conv2D(186, time×8, VALID) → ReLU → Dropout
      → Dense(128) → Dropout → Dense(128) → Dropout → Dense(label_count)
    """
    fw = model_settings['fingerprint_width']
    ft = model_settings['spectrogram_length']
    inp = _flat_input(model_settings)
    x   = tf.keras.layers.Reshape((ft, fw, 1))(inp)

    x   = tf.keras.layers.Conv2D(186, (ft, 8), padding='valid',
              activation='relu',
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros')(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.Flatten()(x)

    for units in (128, 128):
        x = tf.keras.layers.Dense(
                units, activation='relu',
                kernel_initializer=_truncated_normal(0.01),
                bias_initializer='zeros')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

    out = tf.keras.layers.Dense(
              model_settings['label_count'],
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros',
              dtype='float32', name='logits')(x)
    return tf.keras.Model(inp, out, name='low_latency_conv')


# ----------------------------------------------------------------------------
# low_latency_svdf
# ----------------------------------------------------------------------------

class SVDFLayer(tf.keras.layers.Layer):
    """Singular Value Decomposition Filter layer for keyword detection.

    Implements the SVDF operation from:
      'Compressing Deep Neural Networks using a Rank-Constrained Topology'
      https://static.googleusercontent.com/media/research.google.com/en//
      pubs/archive/43813.pdf

    The layer decomposes a full-rank weight matrix W (input_time × num_units)
    into two lower-rank factors:
      • weights_frequency : [input_frequency, num_filters]  (spatial filters)
      • weights_time      : [num_filters, input_time]        (temporal filters)

    where num_filters = rank × num_units.  During training the full context
    window (all time frames) is processed in one shot.  A companion class
    SVDFStreamingLayer handles the incremental inference path.

    WHY A CUSTOM LAYER (not a stateless function)
    ─────────────────────────────────────────────
    The original models.py SVDF uses tf.compat.v1.get_variable for both the
    learnable filters *and* the non-trainable runtime memory.  Keras requires
    weights to be declared in build() / add_weight() so they are properly
    tracked by the optimizer, saved in checkpoints, and mirrored across GPUs
    by MirroredStrategy.  A custom Layer is the correct abstraction.

    The streaming memory is *not* stored here — it lives in
    SVDFStreamingLayer, which wraps the trained weights at export time.
    Mixing streaming state with training would break multi-GPU training and
    make batched validation impossible.

    Args:
        num_units:        Number of SVDF output units (1280 in models.py).
        rank:             Rank of the filter decomposition (2 in models.py).
        input_freq_size:  Number of frequency bins per time frame.
        input_time_size:  Number of time frames in the fingerprint.
    """

    def __init__(self, num_units, rank, input_freq_size, input_time_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_units       = num_units
        self.rank            = rank
        self.input_freq_size = input_freq_size
        self.input_time_size = input_time_size
        self.num_filters     = rank * num_units

    def build(self, input_shape):
        # Frequency filters: [input_frequency_size, num_filters]
        self.weights_frequency = self.add_weight(
            name='weights_frequency',
            shape=(self.input_freq_size, self.num_filters),
            initializer=_truncated_normal(0.01),
            trainable=True)
        # Time filters: [num_filters, input_time_size]
        self.weights_time = self.add_weight(
            name='weights_time',
            shape=(self.num_filters, self.input_time_size),
            initializer=_truncated_normal(0.01),
            trainable=True)
        # Bias: [num_units]
        self.svdf_bias = self.add_weight(
            name='bias',
            shape=(self.num_units,),
            initializer='zeros',
            trainable=True)
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Forward pass.

        During both training and inference here we process the full fingerprint
        (all time frames at once).  The streaming inference path with rolling
        memory is handled separately by SVDFStreamingLayer.

        Matches create_low_latency_svdf_model (is_training=True) in models.py:
          1. expand_dims(inputs, 2)            → [batch, T*F, 1]
          2. conv1d(stride=input_freq_size)    → [batch, T, num_filters]
          3. transpose                         → [num_filters, batch, T]
          4. matmul(w_time)                    → [num_filters, batch, 1]
          5. reshape + reduce_sum + transpose  → [batch, num_units]

        Args:
            inputs:   Float tensor [batch, fingerprint_size]
                      = [batch, input_time_size * input_frequency_size]
            training: Bool — controls Dropout in the parent model, not used
                      directly inside this layer.

        Returns:
            Float tensor [batch, num_units] — SVDF activations before bias
            addition and ReLU (those are applied by the parent model).
        """
        # Treat the flat fingerprint as a 1-D signal with 1 input channel.
        # [batch, T*F] → [batch, T*F, 1]
        # This matches models.py:
        #   new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)
        x = tf.expand_dims(inputs, 2)

        # Apply frequency filters via Conv1D with stride = input_freq_size.
        # filter:  [input_freq_size, 1, num_filters]
        # output:  [batch, input_time_size, num_filters]
        # stride = input_freq_size collapses each freq window into one output
        # frame, producing exactly input_time_size output frames.
        # Matches models.py: stride=input_frequency_size, padding='VALID'
        w_freq = tf.expand_dims(self.weights_frequency, 1)  # [F, 1, num_filters]
        activations_time = tf.nn.conv1d(
            input=x,
            filters=w_freq,
            stride=self.input_freq_size,
            padding='VALID')   # [batch, input_time_size, num_filters]

        # Rearrange to [num_filters, batch, input_time_size]
        activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

        # Apply time filters: weights_time [num_filters, input_time_size, 1]
        w_time  = tf.expand_dims(self.weights_time, 2)
        # outputs: [num_filters, batch, 1]
        outputs = tf.matmul(activations_time, w_time)

        # Reshape [num_filters, batch, 1] → [num_units, rank, batch]
        outputs = tf.reshape(outputs, [self.num_units, self.rank, -1])

        # Sum over rank dimension → [num_units, batch]
        units_output = tf.reduce_sum(outputs, axis=1)

        # Transpose → [batch, num_units]
        units_output = tf.transpose(units_output)

        return tf.nn.bias_add(units_output, self.svdf_bias)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            num_units=self.num_units,
            rank=self.rank,
            input_freq_size=self.input_freq_size,
            input_time_size=self.input_time_size,
        ))
        return cfg


class SVDFStreamingLayer(tf.keras.layers.Layer):
    """SVDF layer with rolling context buffer for streaming (inference only).

    This layer wraps the same frequency and time filter weights as SVDFLayer
    but maintains a non-trainable tf.Variable 'memory' that holds the last
    input_time_size frames of frequency-filter activations.  At each call it:
      1. Computes frequency-filter activations for the new frame(s).
      2. Appends them to the right of the memory buffer (oldest frames drop off
         the left).
      3. Applies the time filter to the full buffer.

    This reproduces the runtime-memory logic of the original TF1
    create_low_latency_svdf_model (is_training=False path) exactly.

    Usage: construct this layer from a trained SVDFLayer's weights, then export
    it inside the streaming SavedModel (see build_low_latency_svdf).

    Args:
        svdf_layer:      A built SVDFLayer instance whose weights to reuse.
        batch_size:      Fixed batch size for inference (usually 1).
    """

    def __init__(self, svdf_layer: SVDFLayer, batch_size: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._svdf = svdf_layer
        self._batch_size = batch_size

    def build(self, input_shape):
        # Non-trainable rolling buffer: [num_filters, batch, input_time_size]
        self.memory = self.add_weight(
            name='memory',
            shape=(self._svdf.num_filters,
                   self._batch_size,
                   self._svdf.input_time_size),
            initializer='zeros',
            trainable=False)
        super().build(input_shape)

    def call(self, inputs, num_new_frames=None):
        """Incremental forward pass using the rolling memory buffer.

        Args:
            inputs:         Float tensor [batch, num_new_frames * freq_size].
                            For the first call pass the full fingerprint
                            (num_new_frames = input_time_size); for subsequent
                            streaming calls pass only the new frame(s).
            num_new_frames: Number of new time frames in inputs.  Defaults to
                            input_time_size (full-window, first call).

        Returns:
            Float tensor [batch, num_units] — same shape as SVDFLayer.call().
        """
        if num_new_frames is None:
            num_new_frames = self._svdf.input_time_size

        # Reshape new input to [batch, num_new_frames * freq_size, 1]
        # and apply frequency filters with stride=input_freq_size.
        # Matches the training path in SVDFLayer.call() exactly.
        x = tf.expand_dims(inputs, 2)   # [batch, num_new_frames*freq_size, 1]

        # Apply frequency filters → [batch, num_new_frames, num_filters]
        w_freq   = tf.expand_dims(self._svdf.weights_frequency, 1)
        new_acts = tf.nn.conv1d(
            x, w_freq,
            stride=self._svdf.input_freq_size,
            padding='VALID')

        # Rearrange → [num_filters, batch, num_new_frames]
        new_acts = tf.transpose(new_acts, perm=[2, 0, 1])

        # Update rolling memory: drop oldest, append new activations
        updated_memory = tf.concat(
            [self.memory[:, :, num_new_frames:], new_acts], axis=2)
        self.memory.assign(updated_memory)

        # Apply time filter to the full buffer
        w_time  = tf.expand_dims(self._svdf.weights_time, 2)
        outputs = tf.matmul(updated_memory, w_time)

        # [num_filters, batch, 1] → [num_units, rank, batch]
        outputs = tf.reshape(
            outputs,
            [self._svdf.num_units, self._svdf.rank, -1])

        # Sum rank → [num_units, batch] → [batch, num_units]
        units_output = tf.transpose(tf.reduce_sum(outputs, axis=1))

        return tf.nn.bias_add(units_output, self._svdf.svdf_bias)

    def reset_memory(self):
        """Zeros the rolling buffer (call between utterances at inference)."""
        self.memory.assign(tf.zeros_like(self.memory))


def build_low_latency_svdf(model_settings):
    """Port of create_low_latency_svdf_model as a Keras model.

    Training path (returned model)
    ───────────────────────────────
    fingerprint → SVDFLayer (rank=2, num_units=1280) → ReLU → Dropout
      → Dense(256) → Dropout → Dense(256) → Dropout → Dense(label_count)

    This is identical to the TF1 is_training=True path: the full window is
    processed in one shot with no memory state.

    Inference / streaming path
    ───────────────────────────
    After training call build_svdf_streaming_model(trained_model) to get
    the streaming version with the rolling context buffer.

    Args:
        model_settings: Dict from model_settings_lib.prepare_model_settings().

    Returns:
        Compiled tf.keras.Model for batch training.
    """
    fw = model_settings['fingerprint_width']     # frequency bins per frame
    ft = model_settings['spectrogram_length']    # number of time frames

    inp        = _flat_input(model_settings)
    svdf_layer = SVDFLayer(
        num_units=1280,
        rank=2,
        input_freq_size=fw,
        input_time_size=ft,
        name='svdf')

    # SVDFLayer returns [batch, num_units] with bias already added.
    # Training path: full window processed in one shot (no memory state).
    x = svdf_layer(inp)
    x = tf.keras.layers.Activation('relu', name='svdf_relu')(x)
    x = tf.keras.layers.Dropout(0.5, name='svdf_dropout')(x)

    # Two fully-connected layers matching the original models.py topology.
    x = tf.keras.layers.Dense(
            256, activation='relu',
            kernel_initializer=_truncated_normal(0.01),
            bias_initializer='zeros', name='fc1')(x)
    x = tf.keras.layers.Dropout(0.5, name='fc1_dropout')(x)

    x = tf.keras.layers.Dense(
            256, activation='relu',
            kernel_initializer=_truncated_normal(0.01),
            bias_initializer='zeros', name='fc2')(x)
    x = tf.keras.layers.Dropout(0.5, name='fc2_dropout')(x)

    # Output layer pinned to float32 for numerical stability with mixed
    # precision — same reason as all other architectures.
    out = tf.keras.layers.Dense(
              model_settings['label_count'],
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros',
              dtype='float32', name='logits')(x)

    return tf.keras.Model(inp, out, name='low_latency_svdf')


def build_svdf_streaming_model(trained_model, batch_size=1):
    """Builds a streaming inference model from a trained low_latency_svdf model.

    Extracts the trained SVDFLayer weights and constructs a new model that uses
    SVDFStreamingLayer (rolling context buffer) in place of SVDFLayer.

    Call this after training to get a model suitable for freeze / TFLite export
    for streaming keyword detection on MCUs.

    Args:
        trained_model: A trained tf.keras.Model built by build_low_latency_svdf.
        batch_size:    Inference batch size (usually 1 for MCU streaming).

    Returns:
        tf.keras.Model with streaming SVDFStreamingLayer.
    """
    # Find the SVDFLayer in the trained model.
    svdf_layer = next(
        l for l in trained_model.layers if isinstance(l, SVDFLayer))

    # Build the streaming variant with the same trained weights.
    streaming_svdf = SVDFStreamingLayer(svdf_layer, batch_size=batch_size,
                                        name='svdf_streaming')

    # Re-assemble the Dense head using the trained weights.
    dense_layers = [l for l in trained_model.layers
                    if isinstance(l, tf.keras.layers.Dense)]

    inp = tf.keras.Input(
        shape=(svdf_layer.input_freq_size * svdf_layer.input_time_size,),
        batch_size=batch_size,
        name='fingerprint_input')
    x   = streaming_svdf(inp)
    x   = tf.keras.layers.Activation('relu')(x)

    for dense in dense_layers:
        x = tf.keras.layers.Dense(
                dense.units,
                activation=dense.activation,
                weights=dense.get_weights(),
                trainable=False)(x)

    streaming_model = tf.keras.Model(inp, x, name='low_latency_svdf_streaming')
    return streaming_model


# ----------------------------------------------------------------------------
# tiny_conv
# ----------------------------------------------------------------------------

def build_tiny_conv(model_settings):
    """Port of create_tiny_conv_model.

    fingerprint → reshape(time, freq, 1)
      → Conv2D(8, 10×8, stride 2×2, SAME) → ReLU → Dropout(0.5)
      → Flatten → Dense(label_count)
    """
    fw  = model_settings['fingerprint_width']
    ft  = model_settings['spectrogram_length']
    inp = _flat_input(model_settings)
    x   = tf.keras.layers.Reshape((ft, fw, 1))(inp)
    x   = tf.keras.layers.Conv2D(8, (10, 8), strides=(2, 2), padding='same',
              activation='relu',
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros')(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(
              model_settings['label_count'],
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros',
              dtype='float32', name='logits')(x)
    return tf.keras.Model(inp, out, name='tiny_conv')


# ----------------------------------------------------------------------------
# tiny_embedding_conv
# ----------------------------------------------------------------------------

def build_tiny_embedding_conv(model_settings):
    """Port of create_tiny_embedding_conv_model.

    fingerprint → reshape(time, freq, 1)
      → Conv2D(8, 10×8, stride 2×2, SAME) → ReLU → Dropout(0.5)
      → Conv2D(8, 10×8, stride 8×8, SAME) → ReLU → Dropout(0.5)
      → Flatten → Dense(label_count)
    """
    fw  = model_settings['fingerprint_width']
    ft  = model_settings['spectrogram_length']
    inp = _flat_input(model_settings)
    x   = tf.keras.layers.Reshape((ft, fw, 1))(inp)
    x   = tf.keras.layers.Conv2D(8, (10, 8), strides=(2, 2), padding='same',
              activation='relu',
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros')(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.Conv2D(8, (10, 8), strides=(8, 8), padding='same',
              activation='relu',
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros')(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(
              model_settings['label_count'],
              kernel_initializer=_truncated_normal(0.01),
              bias_initializer='zeros',
              dtype='float32', name='logits')(x)
    return tf.keras.Model(inp, out, name='tiny_embedding_conv')


# Registry — maps CLI flag to builder function.
MODEL_BUILDERS = {
    'single_fc':            build_single_fc,
    'conv':                 build_conv,
    'low_latency_conv':     build_low_latency_conv,
    'low_latency_svdf':     build_low_latency_svdf,
    'tiny_conv':            build_tiny_conv,
    'tiny_embedding_conv':  build_tiny_embedding_conv,
}


def build_model(architecture, model_settings):
    """Instantiates and compiles the requested architecture.

    The final Dense layer is always dtype='float32' to keep cross-entropy
    loss numerically stable when the global policy is 'mixed_float16'.
    jit_compile=True enables XLA fusion of the training step for ~10–30%
    throughput improvement on GPU.

    Optimizer note
    ──────────────
    When mixed_float16 is active, Keras automatically wraps Adam in a
    LossScaleOptimizer.  The default dynamic loss scaler uses tf.cond
    to check for non-finite gradients, which creates a cross-replica
    synchronisation point inside the XLA-compiled training step.
    MirroredStrategy + jit_compile=True cannot cross that boundary and
    raises "merge_call called while defining a new graph".

    Fix: wrap Adam in a *fixed* LossScaleOptimizer (dynamic=False).
    A fixed scale never calls tf.cond, making it fully compatible with
    jit_compile=True and MirroredStrategy.  initial_scale=2**15 (32768)
    is the standard value for float16 mixed-precision training.

    Args:
        architecture:   Key from MODEL_BUILDERS.
        model_settings: Dict from model_settings_lib.prepare_model_settings().

    Returns:
        Compiled tf.keras.Model.
    """
    model = MODEL_BUILDERS[architecture](model_settings)

    base_opt = tf.keras.optimizers.Adam()
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        # Fixed scale avoids the dynamic tf.cond incompatible with
        # jit_compile=True + MirroredStrategy.
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            base_opt, dynamic=False, initial_scale=2 ** 15)
    else:
        optimizer = base_opt

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        jit_compile=True,
    )
    model.summary()
    return model


# ============================================================================
# Learning-rate schedule
# ============================================================================

def build_lr_schedule(how_many_training_steps, learning_rate,
                       steps_per_epoch):
    """Converts train.py step-based LR stages to Keras epoch callbacks.

    Args:
        how_many_training_steps: Comma-separated step counts per stage.
        learning_rate:           Comma-separated LR values per stage.
        steps_per_epoch:         Gradient steps per epoch.

    Returns:
        (total_epochs, LearningRateScheduler) tuple.
    """
    step_counts = [int(s)   for s in how_many_training_steps.split(',')]
    lr_values   = [float(lr) for lr in learning_rate.split(',')]
    if len(step_counts) != len(lr_values):
        raise ValueError(
            '--how_many_training_steps and --learning_rate must have the '
            'same number of comma-separated values.')

    cumulative = []
    running    = 0
    for steps, lr in zip(step_counts, lr_values):
        running += max(1, math.ceil(steps / steps_per_epoch))
        cumulative.append((running, lr))

    def _schedule(epoch, _lr):
        for boundary, lr in cumulative:
            if epoch < boundary:
                return lr
        return cumulative[-1][1]

    return cumulative[-1][0], tf.keras.callbacks.LearningRateScheduler(
        _schedule, verbose=1)


# ============================================================================
# Plotting helpers
# ============================================================================

def save_training_curves(history, path):
    m = history.history
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, m['loss'],     label='loss')
    plt.plot(history.epoch, m['val_loss'], label='val_loss')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, 100 * np.array(m['accuracy']),     label='acc')
    plt.plot(history.epoch, 100 * np.array(m['val_accuracy']), label='val_acc')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Accuracy [%]')
    plt.tight_layout(); plt.savefig(path); plt.close()
    print(f'Training curves → {path}')


def save_confusion_matrix(model, test_ds, label_names, path):
    y_pred = tf.argmax(model.predict(test_ds), axis=1)
    y_true = tf.concat([lab for _, lab in test_ds], axis=0)
    cm     = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=label_names, yticklabels=label_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction'); plt.ylabel('Label')
    plt.tight_layout(); plt.savefig(path); plt.close()
    print(f'Confusion matrix → {path}')


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # ── 1. GPU ────────────────────────────────────────────────────────────
    strategy, num_gpus = configure_gpu()

    # ── 2. Model settings (identical computation to train.py) ─────────────
    wanted_words_list = args.wanted_words.split(',')
    words_list        = input_data.prepare_words_list(wanted_words_list)

    model_settings = model_settings_lib.prepare_model_settings(
        label_count       = len(words_list),
        sample_rate       = args.sample_rate,
        clip_duration_ms  = args.clip_duration_ms,
        window_size_ms    = args.window_size_ms,
        window_stride_ms  = args.window_stride,
        feature_bin_count = args.feature_bin_count,
        preprocess        = args.preprocess,
    )
    print(f'\nfingerprint_size={model_settings["fingerprint_size"]}, '
          f'spectrogram_length={model_settings["spectrogram_length"]}, '
          f'fingerprint_width={model_settings["fingerprint_width"]}')

    # ── 3. AudioProcessor ─────────────────────────────────────────────────
    print('\nInitialising AudioProcessor …')
    audio_proc = input_data.AudioProcessor(
        data_url              = args.data_url,
        data_dir              = args.data_dir,
        silence_percentage    = args.silence_percentage,
        unknown_percentage    = args.unknown_percentage,
        wanted_words          = wanted_words_list,
        validation_percentage = args.validation_percentage,
        testing_percentage    = args.testing_percentage,
        model_settings        = model_settings,
        summaries_dir         = args.summaries_dir,
    )

    time_shift_samples = int(
        (args.time_shift_ms * args.sample_rate) / 1000)

    print('\nBuilding tf.data datasets …')
    train_ds = make_dataset(audio_proc, model_settings, 'training',
                            args.batch_size, args.background_frequency,
                            args.background_volume, time_shift_samples,
                            num_gpus, shuffle=True)
    val_ds   = make_dataset(audio_proc, model_settings, 'validation',
                            args.batch_size, 0.0, 0.0, 0,
                            num_gpus, shuffle=False)
    test_ds  = make_dataset(audio_proc, model_settings, 'testing',
                            args.batch_size, 0.0, 0.0, 0,
                            num_gpus, shuffle=False)

    # ── 4. LR schedule ────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(
        audio_proc.set_size('training')
        / (args.batch_size * max(num_gpus, 1)))
    total_epochs, lr_cb = build_lr_schedule(
        args.how_many_training_steps, args.learning_rate, steps_per_epoch)
    print(f'\nTraining for {total_epochs} epochs '
          f'({steps_per_epoch} steps/epoch) …')

    # ── 5. Model (inside strategy scope) ──────────────────────────────────
    os.makedirs(args.train_dir, exist_ok=True)
    ckpt = os.path.join(args.train_dir, 'ckpt-{epoch:02d}.weights.h5')

    with strategy.scope():
        model = build_model(args.model_architecture, model_settings)

    # ── 6. Train ──────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        callbacks=[
            lr_cb,
            tf.keras.callbacks.ModelCheckpoint(
                ckpt, save_weights_only=True, verbose=0),
            tf.keras.callbacks.TensorBoard(log_dir=args.summaries_dir),
        ],
    )

    # ── 7. Evaluate ───────────────────────────────────────────────────────
    print('\nTest set evaluation:')
    model.evaluate(test_ds, return_dict=True)

    # ── 8. Plots ──────────────────────────────────────────────────────────
    os.makedirs(args.plot_dir, exist_ok=True)
    save_training_curves(
        history, os.path.join(args.plot_dir, 'training_curves.png'))
    save_confusion_matrix(
        model, test_ds, words_list,
        os.path.join(args.plot_dir, 'confusion_matrix.png'))

    # ── 9. Save ───────────────────────────────────────────────────────────
    os.makedirs(args.saved_model_dir, exist_ok=True)
    model.save(os.path.join(args.saved_model_dir, 'model.keras'))
    print(f'\nKeras model saved to: '
          f'{os.path.join(args.saved_model_dir, "model.keras")}')

    # For SVDF: also export the streaming variant
    if args.model_architecture == 'low_latency_svdf':
        streaming_dir = os.path.join(args.saved_model_dir, 'streaming')
        streaming     = build_svdf_streaming_model(model, batch_size=1)
        tf.saved_model.save(streaming, streaming_dir)
        print(f'Streaming SVDF model saved to: {streaming_dir}')
        print('Use the streaming model for TFLite / MCU deployment.')

    print('\nNext step: run convert_tflite.py to produce quantised artefacts.')


if __name__ == '__main__':
    main()