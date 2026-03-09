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
"""Input data handling for simple speech recognition.

TF2 refactor — GPU-aware pipeline.

CPU vs GPU split
----------------
Audio decoding (tf.audio.decode_wav) and spectrogram ops (audio_spectrogram,
mfcc) have no GPU kernels and MUST run on CPU.  Everything downstream of the
feature vector — matrix multiplies, activations, gradients — runs on GPU.

This file's responsibility ends at the feature vector.  The tf.data pipeline
produces float32 feature tensors on CPU; train.py prefetches them onto the GPU
via dataset.prefetch(tf.data.AUTOTUNE) with experimental_device set, so the
GPU never stalls waiting for the next batch.

sess compatibility
------------------
All public method signatures retain the `sess` parameter for drop-in
compatibility with existing call sites.  It is accepted and ignored.

@tf.function without input_signature
-------------------------------------
audio_ops.mfcc / audio_spectrogram return tensors with a dynamic time
dimension.  Any @tf.function decorated with input_signature forces TF to
trace symbolically and assigns that dimension 0, making tf.reshape(...,
[fp_size]) raise at trace time.  The fix is plain @tf.function (no
input_signature) so TF traces lazily with real shapes on the first call.
"""
import hashlib
import math
import os
import os.path
import random
import re
import sys
import tarfile
import urllib

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

try:
    from tensorflow.lite.experimental.microfrontend.python.ops import (
        audio_microfrontend_op as frontend_op,
    )
except ImportError:
    frontend_op = None

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1
SILENCE_LABEL     = '_silence_'
SILENCE_INDEX     = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def prepare_words_list(wanted_words):
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    base_name        = os.path.basename(filename)
    hash_name        = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash  = (
        (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
        * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        return 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        return 'testing'
    return 'training'


def load_wav_file(filename):
    raw     = io_ops.read_file(filename)
    decoded = tf.audio.decode_wav(raw, desired_channels=1)
    return decoded.audio.numpy().flatten()


def save_wav_file(filename, wav_data, sample_rate):
    wav_tensor = tf.constant(np.reshape(wav_data, (-1, 1)), dtype=tf.float32)
    encoded    = tf.audio.encode_wav(wav_tensor, sample_rate)
    io_ops.write_file(filename, encoded)


def get_features_range(model_settings):
    mode = model_settings['preprocess']
    if mode == 'average': return 0.0,   127.5
    if mode == 'mfcc':    return -247.0, 30.0
    if mode == 'micro':   return 0.0,    26.0
    raise Exception('Unknown preprocess mode "%s"' % mode)


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------

class AudioProcessor(object):
    """Loads, partitions, and prepares audio training data.

    Feature extraction runs on CPU (audio ops have no GPU kernels).
    The resulting float32 tensors are consumed by a tf.data pipeline in
    train.py that transfers them to GPU memory ahead of each training step.
    """

    def __init__(self, data_url, data_dir, silence_percentage,
                 unknown_percentage, wanted_words, validation_percentage,
                 testing_percentage, model_settings, summaries_dir):
        if data_dir:
            self.data_dir = data_dir
            self.maybe_download_and_extract_dataset(data_url, data_dir)
            self.prepare_data_index(
                silence_percentage, unknown_percentage, wanted_words,
                validation_percentage, testing_percentage)
            self.prepare_background_data()
        self.prepare_processing_graph(model_settings, summaries_dir)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        if not data_url:
            return
        if not gfile.Exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not gfile.Exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            try:
                filepath, _ = urllib.request.urlretrieve(
                    data_url, filepath, _progress)
            except Exception:
                tf.get_logger().error(
                    'Failed to download %s to %s', data_url, filepath)
                raise
            print()
            statinfo = os.stat(filepath)
            tf.get_logger().info(
                'Downloaded %s (%d bytes)', filename, statinfo.st_size)
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    # ------------------------------------------------------------------
    # Data index
    # ------------------------------------------------------------------

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                            wanted_words, validation_percentage,
                            testing_percentage):
        random.seed(RANDOM_SEED)
        wanted_words_index = {w: i + 2 for i, w in enumerate(wanted_words)}
        self.data_index  = {'validation': [], 'testing': [], 'training': []}
        unknown_index    = {'validation': [], 'testing': [], 'training': []}
        all_words        = {}

        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word    = word.lower()
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage,
                                   testing_percentage)
            if word in wanted_words_index:
                self.data_index[set_index].append(
                    {'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append(
                    {'label': word, 'file': wav_path})

        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for wanted_word in wanted_words:
            if wanted_word not in all_words:
                raise Exception(
                    'Expected to find ' + wanted_word +
                    ' in labels but only found ' +
                    ', '.join(all_words.keys()))

        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size     = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append(
                    {'label': SILENCE_LABEL, 'file': silence_wav_path})
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(
                unknown_index[set_index][:unknown_size])

        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])

        self.words_list    = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            self.word_to_index[word] = (wanted_words_index[word]
                                        if word in wanted_words_index
                                        else UNKNOWN_WORD_INDEX)
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    # ------------------------------------------------------------------
    # Background data
    # ------------------------------------------------------------------

    def prepare_background_data(self):
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not gfile.Exists(background_dir):
            return self.background_data

        search_path = os.path.join(
            self.data_dir, BACKGROUND_NOISE_DIR_NAME, '*.wav')
        for wav_path in gfile.Glob(search_path):
            raw     = io_ops.read_file(wav_path)
            decoded = tf.audio.decode_wav(raw, desired_channels=1)
            self.background_data.append(decoded.audio.numpy().flatten())

        if not self.background_data:
            raise Exception(
                'No background wav files were found in ' + search_path)

    # ------------------------------------------------------------------
    # Processing graph
    # ------------------------------------------------------------------

    def prepare_processing_graph(self, model_settings, summaries_dir):
        """Builds _run_graph: the CPU-side feature extraction callable.

        Why @tf.function without input_signature
        -----------------------------------------
        audio_ops.mfcc returns a tensor with a dynamic time-dimension.
        input_signature forces symbolic tracing which assigns that dim = 0,
        making any tf.reshape(..., [fp_size]) raise at trace time.
        Plain @tf.function traces lazily on the first *real* call with
        concrete shapes — no reshape failures, full graph-mode speed.

        Why CPU
        -------
        audio_spectrogram and mfcc have no GPU kernels; TF would silently
        fall back to CPU anyway.  Being explicit avoids unnecessary H2D/D2H
        copies of intermediate tensors.  The final float32 feature vector is
        what gets transferred to the GPU by the tf.data prefetch in train.py.
        """
        self._model_settings  = model_settings
        self.summary_writer_  = None
        self.merged_summaries_ = None

        if summaries_dir:
            self.summary_writer_ = tf.summary.create_file_writer(
                os.path.join(summaries_dir, 'data'))

        desired_samples = model_settings['desired_samples']
        window_size     = model_settings['window_size_samples']
        window_stride   = model_settings['window_stride_samples']
        fp_width        = model_settings['fingerprint_width']
        avg_win         = model_settings.get('average_window_width', -1)
        sample_rate     = model_settings['sample_rate']
        preprocess      = model_settings['preprocess']

        self._desired_samples = desired_samples
        self._fp_size         = model_settings['fingerprint_size']

        # All ops in _run_graph are explicitly pinned to CPU because
        # audio_spectrogram / mfcc have no GPU kernels.  The resulting
        # feature tensor is a plain float32 array that the GPU pipeline
        # in train.py can consume with zero-copy prefetch.
        @tf.function
        def _run_graph(wav_path, foreground_volume,
                       time_shift_padding, time_shift_offset,
                       background_data, background_volume):
            with tf.device('/CPU:0'):
                raw         = io_ops.read_file(wav_path)
                wav_decoded = tf.audio.decode_wav(
                    raw, desired_channels=1,
                    desired_samples=desired_samples)

                scaled_foreground = tf.multiply(
                    wav_decoded.audio, foreground_volume)
                padded_foreground = tf.pad(
                    scaled_foreground, time_shift_padding, mode='CONSTANT')
                sliced_foreground = tf.slice(
                    padded_foreground, time_shift_offset,
                    [desired_samples, -1])

                background_mul   = tf.multiply(background_data, background_volume)
                background_clamp = tf.clip_by_value(
                    tf.add(background_mul, sliced_foreground), -1.0, 1.0)

                spectrogram = audio_ops.audio_spectrogram(
                    background_clamp,
                    window_size=window_size,
                    stride=window_stride,
                    magnitude_squared=True)

                if preprocess == 'average':
                    output = tf.nn.pool(
                        input=tf.expand_dims(spectrogram, -1),
                        window_shape=[1, avg_win],
                        strides=[1, avg_win],
                        pooling_type='AVG',
                        padding='SAME')
                elif preprocess == 'mfcc':
                    output = audio_ops.mfcc(
                        spectrogram,
                        wav_decoded.sample_rate,
                        dct_coefficient_count=fp_width)
                elif preprocess == 'micro':
                    if not frontend_op:
                        raise Exception(
                            'Micro frontend op unavailable outside Bazel.')
                    ws_ms       = (window_size   * 1000) / sample_rate
                    wt_ms       = (window_stride * 1000) / sample_rate
                    int16_input = tf.cast(
                        tf.multiply(background_clamp, 32768), tf.int16)
                    micro       = frontend_op.audio_microfrontend(
                        int16_input, sample_rate=sample_rate,
                        window_size=ws_ms, window_step=wt_ms,
                        num_channels=fp_width, out_scale=1,
                        out_type=tf.float32)
                    output = tf.multiply(micro, 10.0 / 256.0)
                else:
                    raise ValueError(
                        'Unknown preprocess mode "%s"' % preprocess)

                # Flatten happens OUTSIDE the tf.function — see module docstring.
                return output

        self._run_graph = _run_graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_size(self, mode):
        return len(self.data_index[mode])

    def get_data(self, how_many, offset, model_settings, background_frequency,
                 background_volume_range, time_shift, mode, sess=None):
        """Return (fingerprints, labels) as numpy arrays.

        Feature extraction runs on CPU (audio ops have no GPU kernels).
        The arrays are consumed by the tf.data GPU pipeline in train.py.
        `sess` is accepted and ignored.

        Returns:
          data   – np.ndarray [sample_count, fingerprint_size]  float32
          labels – np.ndarray [sample_count]                    float64
        """
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        data             = np.zeros((sample_count, model_settings['fingerprint_size']),
                                    dtype=np.float32)
        labels           = np.zeros(sample_count)
        desired_samples  = model_settings['desired_samples']
        use_background   = self.background_data and (mode == 'training')
        pick_deterministically = (mode != 'training')

        for i in range(offset, offset + sample_count):
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]

            # Time shift
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset  = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset  = [-time_shift_amount, 0]

            # Background noise
            if use_background or sample['label'] == SILENCE_LABEL:
                background_index   = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                if len(background_samples) <= desired_samples:
                    raise ValueError(
                        'Background sample is too short! Need more than %d'
                        ' samples but only %d were found' %
                        (desired_samples, len(background_samples)))
                background_offset  = np.random.randint(
                    0, len(background_samples) - desired_samples)
                background_clipped = background_samples[
                    background_offset: background_offset + desired_samples]
                background_reshaped = background_clipped.reshape(
                    [desired_samples, 1])
                if sample['label'] == SILENCE_LABEL:
                    background_volume = np.random.uniform(0, 1)
                elif np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(
                        0, background_volume_range)
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([desired_samples, 1],
                                                dtype=np.float32)
                background_volume   = 0

            foreground_volume = 0 if sample['label'] == SILENCE_LABEL else 1

            # Feature extraction — runs on CPU (see _run_graph).
            output = self._run_graph(
                tf.constant(sample['file']),
                tf.constant(foreground_volume,    dtype=tf.float32),
                tf.constant(time_shift_padding,   dtype=tf.int32),
                tf.constant(time_shift_offset,    dtype=tf.int32),
                tf.constant(background_reshaped,  dtype=tf.float32),
                tf.constant(background_volume,    dtype=tf.float32),
            )

            if self.summary_writer_:
                with self.summary_writer_.as_default():
                    tf.summary.histogram('features', output, step=i)

            # Flatten in NumPy — outside TF tracing, no reshape errors.
            data[i - offset, :]  = output.numpy().flatten()
            labels[i - offset]   = self.word_to_index[sample['label']]

        return data, labels

    def get_features_for_wav(self, wav_filename, model_settings, sess=None):
        """Extract features from a single WAV.  `sess` ignored.

        Returns a list wrapping one numpy array, matching the TF1 return
        format of  sess.run([self.output_], ...).
        """
        desired_samples = self._desired_samples
        output = self._run_graph(
            tf.constant(wav_filename),
            tf.constant(1.0,  dtype=tf.float32),
            tf.constant([[0, 0], [0, 0]], dtype=tf.int32),
            tf.constant([0, 0],           dtype=tf.int32),
            tf.constant(np.zeros([desired_samples, 1], dtype=np.float32),
                        dtype=tf.float32),
            tf.constant(0.0,  dtype=tf.float32),
        )
        # Flatten to [fp_size] in NumPy — outside TF tracing — and wrap in a
        # list to match the TF1 return format: sess.run([self.output_], ...).
        return [output.numpy().flatten().astype(np.float32)]

    def get_unprocessed_data(self, how_many, model_settings, mode):
        """Return raw unaugmented audio samples."""
        candidates      = self.data_index[mode]
        sample_count    = len(candidates) if how_many == -1 else how_many
        desired_samples = model_settings['desired_samples']
        words_list      = self.words_list

        data   = np.zeros((sample_count, desired_samples), dtype=np.float32)
        labels = []

        for i in range(sample_count):
            sample_index = (i if how_many == -1
                            else np.random.randint(len(candidates)))
            sample  = candidates[sample_index]
            raw     = io_ops.read_file(sample['file'])
            decoded = tf.audio.decode_wav(
                raw, desired_channels=1, desired_samples=desired_samples)
            fg_vol  = 0.0 if sample['label'] == SILENCE_LABEL else 1.0
            scaled  = tf.multiply(decoded.audio,
                                   tf.constant(fg_vol, dtype=tf.float32))
            data[i, :]  = scaled.numpy().flatten()
            label_index = self.word_to_index[sample['label']]
            labels.append(words_list[label_index])

        return data, labels