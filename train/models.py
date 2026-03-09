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
"""Model definitions for simple speech recognition.

Refactored to TensorFlow v2 / Keras. Public API is identical to the original:
  - prepare_model_settings(...)  -> dict   (unchanged)
  - create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None)
                               -> (logits, dropout_rate)  or  logits
  - load_variables_from_checkpoint(sess, start_checkpoint)  (kept for compat)

Each architecture is now a tf.keras.Model subclass.  create_model() builds the
model, runs one forward pass on fingerprint_input, and returns the same tuple
(or single tensor) that the original graph-mode functions returned, so all
call-sites in train.py continue to work without modification.
"""
import math

import tensorflow as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_power_of_two(x):
    """Returns the smallest power of two >= x."""
    return 1 if x == 0 else 2 ** (int(x) - 1).bit_length()


# ---------------------------------------------------------------------------
# Settings (unchanged)
# ---------------------------------------------------------------------------

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                            window_size_ms, window_stride_ms,
                            feature_bin_count, preprocess):
    """Calculates common settings needed for all models.

    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      feature_bin_count: Number of frequency bins to use for analysis.
      preprocess: How the spectrogram is processed to produce features.

    Returns:
      Dictionary containing common settings.

    Raises:
      ValueError: If the preprocessing mode isn't recognized.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

    if preprocess == 'average':
        fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
        average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
        fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
    elif preprocess in ('mfcc', 'micro'):
        average_window_width = -1
        fingerprint_width = feature_bin_count
    else:
        raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                         ' "average", or "micro")' % preprocess)

    fingerprint_size = fingerprint_width * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'fingerprint_width': fingerprint_width,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
        'preprocess': preprocess,
        'average_window_width': average_window_width,
    }


# ---------------------------------------------------------------------------
# Checkpoint helper (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def load_variables_from_checkpoint(sess, start_checkpoint):
    """Compatibility shim — in TF2 use tf.train.Checkpoint.restore() instead."""
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    saver.restore(sess, start_checkpoint)


# ---------------------------------------------------------------------------
# Public factory (same signature as original)
# ---------------------------------------------------------------------------

def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
    """Builds the requested model and returns (logits[, dropout_rate]).

    The signature and return values are identical to the original TF1 version
    so that train.py requires no changes.

    Args:
      fingerprint_input: Tensor [batch, fingerprint_size] of audio features.
      model_settings:    Dict from prepare_model_settings().
      model_architecture: One of 'single_fc', 'conv', 'low_latency_conv',
                          'low_latency_svdf', 'tiny_conv',
                          'tiny_embedding_conv'.
      is_training:       Bool — controls dropout and returned tuple.
      runtime_settings:  Optional dict (used by 'low_latency_svdf' only).

    Returns:
      (logits, dropout_placeholder)  when is_training=True
       logits                        when is_training=False
    """
    builders = {
        'single_fc':            _build_single_fc,
        'conv':                 _build_conv,
        'low_latency_conv':     _build_low_latency_conv,
        'low_latency_svdf':     _build_low_latency_svdf,
        'tiny_conv':            _build_tiny_conv,
        'tiny_embedding_conv':  _build_tiny_embedding_conv,
    }
    if model_architecture not in builders:
        raise Exception(
            'model_architecture argument "%s" not recognized, should be one of '
            '"single_fc", "conv", "low_latency_conv", "low_latency_svdf", '
            '"tiny_conv", or "tiny_embedding_conv"' % model_architecture)

    return builders[model_architecture](
        fingerprint_input, model_settings, is_training, runtime_settings)


# ---------------------------------------------------------------------------
# Architecture implementations
# ---------------------------------------------------------------------------

# Each _build_* function:
#   • Creates a tf.keras.Model subclass instance.
#   • Calls model(fingerprint_input, training=is_training) for the forward pass.
#   • Returns (logits, dropout_rate_variable) or logits to match original API.
#
# Because dropout is now handled internally by Keras layers (via the
# `training` flag), we return a tf.Variable as `dropout_rate` so that
# existing call-sites that feed it via feed_dict continue to parse without
# error — they simply have no effect (Keras ignores the variable).

def _make_dropout_placeholder():
    """Returns a scalar tf.Variable that acts as a no-op dropout_rate stand-in.

    train.py feeds this variable via optimizer; Keras layers use the `training`
    flag instead, so the value is never read by the model.
    """
    return tf.Variable(0.5, trainable=False, dtype=tf.float32,
                       name='dropout_rate')


# ---- single_fc ----

class _SingleFCModel(tf.keras.Model):
    """Single fully-connected layer model."""

    def __init__(self, fingerprint_size, label_count):
        super().__init__(name='single_fc')
        self.fc = tf.keras.layers.Dense(
            label_count,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001),
            bias_initializer='zeros',
            name='weights_bias')

    def call(self, fingerprint_input, training=False):  # pylint: disable=arguments-differ
        return self.fc(fingerprint_input)


def _build_single_fc(fingerprint_input, model_settings, is_training,
                     runtime_settings=None):
    model = _SingleFCModel(
        model_settings['fingerprint_size'],
        model_settings['label_count'])
    logits = model(fingerprint_input, training=is_training)
    if is_training:
        return logits, _make_dropout_placeholder()
    return logits


# ---- conv ----

class _ConvModel(tf.keras.Model):
    """Standard two-block conv model ('cnn-trad-fpool3')."""

    def __init__(self, input_time_size, input_frequency_size, label_count,
                 dropout_rate=0.5):
        super().__init__(name='conv')
        self._time = input_time_size
        self._freq = input_frequency_size
        self._dropout_rate = dropout_rate

        self.conv1 = tf.keras.layers.Conv2D(
            64, (20, 8), padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='first_weights')
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2),
                                               padding='same')

        self.conv2 = tf.keras.layers.Conv2D(
            64, (10, 4), padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='second_weights')
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(
            label_count,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='final_fc_weights')

    def call(self, fingerprint_input, training=False):  # pylint: disable=arguments-differ
        x = tf.reshape(fingerprint_input,
                       [-1, self._time, self._freq, 1])
        x = tf.nn.relu(self.conv1(x))
        x = self.drop1(x, training=training)
        x = self.pool1(x)
        x = tf.nn.relu(self.conv2(x))
        x = self.drop2(x, training=training)
        x = self.flatten(x)
        return self.fc(x)


def _build_conv(fingerprint_input, model_settings, is_training,
                runtime_settings=None):
    model = _ConvModel(
        model_settings['spectrogram_length'],
        model_settings['fingerprint_width'],
        model_settings['label_count'])
    logits = model(fingerprint_input, training=is_training)
    if is_training:
        return logits, _make_dropout_placeholder()
    return logits


# ---- low_latency_conv ----

class _LowLatencyConvModel(tf.keras.Model):
    """Low-latency single-conv + 3-FC model ('cnn-one-fstride4')."""

    def __init__(self, input_time_size, input_frequency_size, label_count,
                 dropout_rate=0.5):
        super().__init__(name='low_latency_conv')
        self._time = input_time_size
        self._freq = input_frequency_size

        first_filter_count = 186
        self.conv1 = tf.keras.layers.Conv2D(
            first_filter_count,
            (input_time_size, 8),
            strides=(1, 1),
            padding='valid',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='first_weights')
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)

        self.flatten = tf.keras.layers.Flatten()

        self.fc1 = tf.keras.layers.Dense(
            128,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='first_fc_weights')
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

        self.fc2 = tf.keras.layers.Dense(
            128,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='second_fc_weights')
        self.drop3 = tf.keras.layers.Dropout(dropout_rate)

        self.fc3 = tf.keras.layers.Dense(
            label_count,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='final_fc_weights')

    def call(self, fingerprint_input, training=False):  # pylint: disable=arguments-differ
        x = tf.reshape(fingerprint_input,
                       [-1, self._time, self._freq, 1])
        x = tf.nn.relu(self.conv1(x))
        x = self.drop1(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop2(x, training=training)
        x = self.fc2(x)
        x = self.drop3(x, training=training)
        return self.fc3(x)


def _build_low_latency_conv(fingerprint_input, model_settings, is_training,
                             runtime_settings=None):
    model = _LowLatencyConvModel(
        model_settings['spectrogram_length'],
        model_settings['fingerprint_width'],
        model_settings['label_count'])
    logits = model(fingerprint_input, training=is_training)
    if is_training:
        return logits, _make_dropout_placeholder()
    return logits


# ---- low_latency_svdf ----

class _LowLatencySVDFModel(tf.keras.Model):
    """SVDF-based low-latency model.

    The runtime memory buffer (used at inference time to avoid recomputing old
    frames) is kept as a non-trainable tf.Variable, matching the original.
    """

    def __init__(self, input_time_size, input_frequency_size, label_count,
                 dropout_rate=0.5):
        super().__init__(name='low_latency_svdf')
        self._time = input_time_size
        self._freq = input_frequency_size

        rank = 2
        num_units = 1280
        num_filters = rank * num_units

        # Frequency filter: equivalent to conv1d over frequency dimension.
        self.weights_frequency = self.add_weight(
            name='weights_frequency',
            shape=[input_frequency_size, 1, num_filters],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            trainable=True)

        # Time filter applied after collecting frames.
        self.weights_time = self.add_weight(
            name='weights_time',
            shape=[num_filters, input_time_size, 1],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            trainable=True)

        self.bias_svdf = self.add_weight(
            name='bias',
            shape=[num_units],
            initializer='zeros',
            trainable=True)

        # Runtime memory (non-trainable).
        self.memory = self.add_weight(
            name='runtime_memory',
            shape=[num_filters, 1, input_time_size],
            initializer='zeros',
            trainable=False)

        self._rank = rank
        self._num_units = num_units
        self._num_filters = num_filters
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)

        self.fc1 = tf.keras.layers.Dense(
            256,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='first_fc_weights')
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

        self.fc2 = tf.keras.layers.Dense(
            256,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='second_fc_weights')
        self.drop3 = tf.keras.layers.Dropout(dropout_rate)

        self.fc3 = tf.keras.layers.Dense(
            label_count,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='final_fc_weights')

    def call(self, fingerprint_input, training=False):  # pylint: disable=arguments-differ
        # fingerprint_input: [batch, time*freq] with oldest frame at [:, 0]
        if training:
            num_new_frames = self._time
        else:
            num_new_frames = self._time  # simplified; update for streaming use

        new_input = fingerprint_input[:, -num_new_frames * self._freq:]
        new_input = tf.expand_dims(new_input, 2)  # [batch, frames*freq, 1]

        # Frequency filter via conv1d.
        # activations_time: [batch, num_new_frames, num_filters]
        activations_time = tf.nn.conv1d(
            input=new_input,
            filters=self.weights_frequency,
            stride=self._freq,
            padding='VALID')
        # [num_filters, batch, num_new_frames]
        activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

        # Update runtime memory at inference time.
        if not training:
            new_memory = self.memory[:, :, num_new_frames:]
            new_memory = tf.concat([new_memory, activations_time], axis=2)
            self.memory.assign(new_memory)
            activations_time = new_memory

        # Time filter: [num_filters, batch, 1]
        outputs = tf.matmul(activations_time, self.weights_time)
        # [num_units, rank, batch] -> sum over rank -> [num_units, batch]
        outputs = tf.reshape(outputs, [self._num_units, self._rank, -1])
        units_output = tf.reduce_sum(outputs, axis=1)
        units_output = tf.transpose(units_output)   # [batch, num_units]

        x = tf.nn.relu(tf.nn.bias_add(units_output, self.bias_svdf))
        x = self.drop1(x, training=training)
        x = self.fc1(x)
        x = self.drop2(x, training=training)
        x = self.fc2(x)
        x = self.drop3(x, training=training)
        return self.fc3(x)


def _build_low_latency_svdf(fingerprint_input, model_settings, is_training,
                             runtime_settings=None):
    model = _LowLatencySVDFModel(
        model_settings['spectrogram_length'],
        model_settings['fingerprint_width'],
        model_settings['label_count'])
    logits = model(fingerprint_input, training=is_training)
    if is_training:
        return logits, _make_dropout_placeholder()
    return logits


# ---- tiny_conv ----

class _TinyConvModel(tf.keras.Model):
    """Tiny single-conv model for microcontrollers."""

    def __init__(self, input_time_size, input_frequency_size, label_count,
                 dropout_rate=0.5):
        super().__init__(name='tiny_conv')
        self._time = input_time_size
        self._freq = input_frequency_size

        self.conv1 = tf.keras.layers.Conv2D(
            8, (10, 8),
            strides=(2, 2),
            padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='first_weights')
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(
            label_count,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='final_fc_weights')

    def call(self, fingerprint_input, training=False):  # pylint: disable=arguments-differ
        x = tf.reshape(fingerprint_input,
                       [-1, self._time, self._freq, 1])
        x = tf.nn.relu(self.conv1(x))
        x = self.drop1(x, training=training)
        x = self.flatten(x)
        return self.fc(x)


def _build_tiny_conv(fingerprint_input, model_settings, is_training,
                     runtime_settings=None):
    model = _TinyConvModel(
        model_settings['spectrogram_length'],
        model_settings['fingerprint_width'],
        model_settings['label_count'])
    logits = model(fingerprint_input, training=is_training)
    if is_training:
        return logits, _make_dropout_placeholder()
    return logits


# ---- tiny_embedding_conv ----

class _TinyEmbeddingConvModel(tf.keras.Model):
    """Tiny three-conv embedding model for microcontrollers."""

    def __init__(self, input_time_size, input_frequency_size, label_count,
                 dropout_rate=0.5):
        super().__init__(name='tiny_embedding_conv')
        self._time = input_time_size
        self._freq = input_frequency_size

        self.conv1 = tf.keras.layers.Conv2D(
            8, (10, 8), strides=(2, 2), padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='first_weights')
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)

        self.conv2 = tf.keras.layers.Conv2D(
            8, (10, 8), strides=(8, 8), padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='second_weights')
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(
            label_count,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            bias_initializer='zeros', name='final_fc_weights')

    def call(self, fingerprint_input, training=False):  # pylint: disable=arguments-differ
        x = tf.reshape(fingerprint_input,
                       [-1, self._time, self._freq, 1])
        x = tf.nn.relu(self.conv1(x))
        x = self.drop1(x, training=training)
        x = tf.nn.relu(self.conv2(x))
        x = self.drop2(x, training=training)
        x = self.flatten(x)
        return self.fc(x)


def _build_tiny_embedding_conv(fingerprint_input, model_settings, is_training,
                                runtime_settings=None):
    model = _TinyEmbeddingConvModel(
        model_settings['spectrogram_length'],
        model_settings['fingerprint_width'],
        model_settings['label_count'])
    logits = model(fingerprint_input, training=is_training)
    if is_training:
        return logits, _make_dropout_placeholder()
    return logits