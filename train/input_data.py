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

TF2 + CPU-optimised version.  Public API is identical to the original TF1
file — all method signatures, including the unused `sess` parameter, are
preserved exactly.

Bug fix vs previous version
----------------------------
Defining a `@tf.function` inside `get_data()` caused TF to re-trace on every
call with wrong static shapes, producing the error:
  "Incompatible shapes at component 0: expected [?,0] but got [100,1960]"

The fix is to compile `_process_one` **once** in `prepare_processing_graph()`
with an explicit `input_signature`, store it as `self._process_one`, and reuse
it across all `get_data()` calls.  Background augmentation parameters that
change per-call (frequency, volume, time-shift) are fed as scalar tensors at
call time rather than baked into the closure.
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
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


# ---------------------------------------------------------------------------
# Module-level helpers  (signatures unchanged)
# ---------------------------------------------------------------------------

def prepare_words_list(wanted_words):
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = (
        (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
        * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    )
    if percentage_hash < validation_percentage:
        return 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        return 'testing'
    return 'training'


def load_wav_file(filename):
    raw = io_ops.read_file(filename)
    decoded = tf.audio.decode_wav(raw, desired_channels=1)
    return decoded.audio.numpy().flatten()


def save_wav_file(filename, wav_data, sample_rate):
    wav_tensor = tf.constant(np.reshape(wav_data, (-1, 1)), dtype=tf.float32)
    encoded = tf.audio.encode_wav(wav_tensor, sample_rate)
    io_ops.write_file(filename, encoded)


def get_features_range(model_settings):
    mode = model_settings['preprocess']
    if mode == 'average':
        return 0.0, 127.5
    elif mode == 'mfcc':
        return -247.0, 30.0
    elif mode == 'micro':
        return 0.0, 26.0
    raise Exception('Unknown preprocess mode "%s"' % mode)


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------

class AudioProcessor:
    """Handles loading, partitioning, and preparing audio training data."""

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
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' % (
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
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index   = {'validation': [], 'testing': [], 'training': []}
        all_words = {}

        for wav_path in gfile.Glob(os.path.join(self.data_dir, '*', '*.wav')):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            split = which_set(wav_path, validation_percentage,
                               testing_percentage)
            if word in wanted_words_index:
                self.data_index[split].append({'label': word, 'file': wav_path})
            else:
                unknown_index[split].append({'label': word, 'file': wav_path})

        if not all_words:
            raise Exception('No .wavs found at ' +
                            os.path.join(self.data_dir, '*', '*.wav'))
        for w in wanted_words:
            if w not in all_words:
                raise Exception('Expected to find "%s" but only found: %s'
                                % (w, ', '.join(all_words)))

        silence_wav_path = self.data_index['training'][0]['file']
        for split in ('validation', 'testing', 'training'):
            n = len(self.data_index[split])
            for _ in range(int(math.ceil(n * silence_percentage / 100))):
                self.data_index[split].append(
                    {'label': SILENCE_LABEL, 'file': silence_wav_path})
            random.shuffle(unknown_index[split])
            unk_n = int(math.ceil(n * unknown_percentage / 100))
            self.data_index[split].extend(unknown_index[split][:unk_n])
            random.shuffle(self.data_index[split])

        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {
            w: (wanted_words_index[w] if w in wanted_words_index
                else UNKNOWN_WORD_INDEX)
            for w in all_words
        }
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    # ------------------------------------------------------------------
    # Background data
    # ------------------------------------------------------------------

    def prepare_background_data(self):
        self.background_data = []
        self._background_tensor = None  # built lazily once desired_samples is known

        bg_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not gfile.Exists(bg_dir):
            return

        search = os.path.join(bg_dir, '*.wav')
        for wav_path in gfile.Glob(search):
            raw = io_ops.read_file(wav_path)
            decoded = tf.audio.decode_wav(raw, desired_channels=1)
            self.background_data.append(decoded.audio.numpy().flatten())

        if not self.background_data:
            raise Exception('No background wavs in ' + search)

    def _ensure_background_tensor(self, desired_samples):
        """Build (once) the [N, desired_samples] background constant tensor."""
        if self._background_tensor is not None:
            return
        if not self.background_data:
            self._background_tensor = tf.zeros(
                [1, desired_samples], dtype=tf.float32)
            return
        clips = []
        for bg in self.background_data:
            if len(bg) < desired_samples:
                raise ValueError(
                    'Background clip too short (%d < %d samples)'
                    % (len(bg), desired_samples))
            clips.append(bg[:desired_samples].astype(np.float32))
        self._background_tensor = tf.constant(
            np.stack(clips), dtype=tf.float32)   # [N_clips, desired_samples]

    # ------------------------------------------------------------------
    # Processing graph  — compiled ONCE, reused everywhere
    # ------------------------------------------------------------------

    def prepare_processing_graph(self, model_settings, summaries_dir):
        """Compile _process_audio and _process_one tf.functions once.

        KEY FIX: both functions are built here with explicit input_signature
        so TF traces them exactly once with the correct concrete shapes.
        Redefining a @tf.function inside get_data() caused shape-inference
        failures because TF re-traced with symbolic [None] shapes on every
        Python call.
        """
        self._model_settings = model_settings

        self.summary_writer_ = None
        if summaries_dir:
            self.summary_writer_ = tf.summary.create_file_writer(
                os.path.join(summaries_dir, 'data'))

        preprocess      = model_settings['preprocess']
        desired_samples = model_settings['desired_samples']
        window_size     = model_settings['window_size_samples']
        window_stride   = model_settings['window_stride_samples']
        fp_width        = model_settings['fingerprint_width']
        fp_size         = model_settings['fingerprint_size']
        avg_win         = model_settings.get('average_window_width', -1)
        sample_rate     = model_settings['sample_rate']

        # ---- _process_audio: augment + feature-extract one clip ----------
        #
        # input_signature pins every tensor shape so TF traces exactly once.
        @tf.function(input_signature=[
            tf.TensorSpec([desired_samples, 1], tf.float32),   # foreground
            tf.TensorSpec([desired_samples, 1], tf.float32),   # background
            tf.TensorSpec([], tf.float32),                     # bg_volume
            tf.TensorSpec([], tf.float32),                     # fg_volume
            tf.TensorSpec([2, 2], tf.int32),                   # time_shift_padding
            tf.TensorSpec([2],    tf.int32),                   # time_shift_offset
        ])
        def _process_audio(foreground, background_data,
                            background_volume, foreground_volume,
                            time_shift_padding, time_shift_offset):
            scaled = tf.multiply(foreground, foreground_volume)
            padded = tf.pad(scaled, time_shift_padding, mode='CONSTANT')
            sliced = tf.slice(padded, time_shift_offset, [desired_samples, -1])
            bg_mul = tf.multiply(background_data, background_volume)
            mixed = tf.clip_by_value(tf.add(bg_mul, sliced), -1.0, 1.0)

            spectrogram = audio_ops.audio_spectrogram(
                mixed, window_size=window_size,
                stride=window_stride, magnitude_squared=True)

            if preprocess == 'average':
                output = tf.nn.pool(
                    input=tf.expand_dims(spectrogram, -1),
                    window_shape=[1, avg_win],
                    strides=[1, avg_win],
                    pooling_type='AVG', padding='SAME')
            elif preprocess == 'mfcc':
                output = audio_ops.mfcc(
                    spectrogram, sample_rate,
                    dct_coefficient_count=fp_width)
            elif preprocess == 'micro':
                if not frontend_op:
                    raise Exception(
                        'Micro frontend op unavailable outside Bazel.')
                ws_ms = (window_size   * 1000) / sample_rate
                wt_ms = (window_stride * 1000) / sample_rate
                i16   = tf.cast(tf.multiply(mixed, 32768), tf.int16)
                mf    = frontend_op.audio_microfrontend(
                    i16, sample_rate=sample_rate,
                    window_size=ws_ms, window_step=wt_ms,
                    num_channels=fp_width, out_scale=1,
                    out_type=tf.float32)
                output = tf.multiply(mf, 10.0 / 256.0)
            else:
                raise ValueError('Unknown preprocess mode "%s"' % preprocess)

            # Returns a flat vector of known size [fp_size].
            return tf.reshape(output, [fp_size])

        self._process_audio = _process_audio
        self._fp_size = fp_size
        self._desired_samples = desired_samples

        # ---- _process_one: load + augment one (path, label, flags) row ---
        #
        # augmentation scalars (bg_frequency, bg_volume_range, time_shift,
        # use_background) vary per get_data() call, so they are passed as
        # scalar tensors rather than Python-level closure variables.  This
        # lets us reuse ONE compiled function for training, validation, and
        # testing without retracing.
        #
        # Signature:
        #   wav_path        tf.string  scalar
        #   label           tf.int32   scalar
        #   silence_flag    tf.int32   scalar  (1 = silence sample)
        #   bg_tensor       tf.float32 [N_bg, desired_samples]
        #   bg_frequency    tf.float32 scalar
        #   bg_volume_range tf.float32 scalar
        #   time_shift      tf.int32   scalar
        #   use_background  tf.bool    scalar
        #
        # Returns: (features [fp_size], label scalar int32)
        #
        # NOTE: bg_tensor shape has a dynamic first dim (N_bg) so we leave
        # that axis as None in the TensorSpec.

        @tf.function(input_signature=[
            tf.TensorSpec([], tf.string),                           # wav_path
            tf.TensorSpec([], tf.int32),                            # label
            tf.TensorSpec([], tf.int32),                            # silence_flag
            tf.TensorSpec([None, desired_samples], tf.float32),     # bg_tensor
            tf.TensorSpec([], tf.float32),                          # bg_frequency
            tf.TensorSpec([], tf.float32),                          # bg_volume_range
            tf.TensorSpec([], tf.int32),                            # time_shift
            tf.TensorSpec([], tf.bool),                             # use_background
        ])
        def _process_one(wav_path, label, silence_flag,
                         bg_tensor, bg_frequency, bg_volume_range,
                         time_shift, use_background):
            # Load wav
            audio, _ = tf.audio.decode_wav(
                tf.io.read_file(wav_path),
                desired_channels=1, desired_samples=desired_samples)
            # audio: [desired_samples, 1]

            # Time shift
            shift = tf.cond(
                tf.greater(time_shift, 0),
                lambda: tf.random.uniform(
                    [], -time_shift, time_shift, dtype=tf.int32),
                lambda: tf.constant(0, tf.int32))
            pad_l   = tf.maximum(shift, 0)
            pad_r   = tf.maximum(-shift, 0)
            padding = tf.stack([[pad_l, pad_r], [0, 0]])   # [2,2]
            t_off   = tf.stack([pad_r, 0])                 # [2]

            # Background noise
            n_bg    = tf.shape(bg_tensor)[0]
            bg_idx  = tf.random.uniform([], 0, n_bg, dtype=tf.int32)
            bg_clip = tf.reshape(bg_tensor[bg_idx], [desired_samples, 1])

            bg_vol = tf.cond(
                use_background,
                true_fn=lambda: tf.cond(
                    tf.equal(silence_flag, 1),
                    true_fn=lambda: tf.random.uniform([], 0.0, 1.0),
                    false_fn=lambda: tf.cond(
                        tf.less(tf.random.uniform([]), bg_frequency),
                        true_fn=lambda: tf.random.uniform(
                            [], 0.0, bg_volume_range),
                        false_fn=lambda: tf.constant(0.0))),
                false_fn=lambda: tf.constant(0.0))

            fg_vol = tf.cond(
                tf.equal(silence_flag, 1),
                lambda: tf.constant(0.0),
                lambda: tf.constant(1.0))

            features = _process_audio(
                audio, bg_clip, bg_vol, fg_vol, padding, t_off)
            return features, tf.cast(label, tf.int32)

        self._process_one = _process_one

        # ---- _load_one: raw unaugmented load for get_unprocessed_data ----
        @tf.function(input_signature=[
            tf.TensorSpec([], tf.string),
            tf.TensorSpec([], tf.int32),
        ])
        def _load_one(wav_path, silence_flag):
            audio, _ = tf.audio.decode_wav(
                tf.io.read_file(wav_path),
                desired_channels=1, desired_samples=desired_samples)
            vol = tf.cond(tf.equal(silence_flag, 1),
                          lambda: tf.constant(0.0),
                          lambda: tf.constant(1.0))
            return tf.reshape(audio * vol, [desired_samples])

        self._load_one = _load_one

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_size(self, mode):
        return len(self.data_index[mode])

    # ------------------------------------------------------------------

    def get_data(self, how_many, offset, model_settings, background_frequency,
                 background_volume_range, time_shift, mode, sess=None):
        """Return (fingerprints, labels) via a parallel tf.data pipeline.

        Signature identical to the original TF1 version.  `sess` is ignored.

        Returns:
          data   – np.ndarray [sample_count, fingerprint_size] float32
          labels – np.ndarray [sample_count] int32
        """
        candidates      = self.data_index[mode]
        desired_samples = self._desired_samples
        fp_size         = self._fp_size

        self._ensure_background_tensor(desired_samples)

        # Determine which samples to process
        if how_many == -1:
            sample_count = len(candidates)
            indices = list(range(len(candidates)))
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
            if mode == 'training':
                indices = np.random.randint(
                    0, len(candidates), size=sample_count).tolist()
            else:
                indices = list(range(offset, offset + sample_count))

        if sample_count == 0:
            return (np.zeros((0, fp_size), dtype=np.float32),
                    np.zeros(0, dtype=np.int32))

        files         = [candidates[i]['file']  for i in indices]
        word_indices  = [self.word_to_index[candidates[i]['label']]
                         for i in indices]
        silence_flags = [1 if candidates[i]['label'] == SILENCE_LABEL else 0
                         for i in indices]

        use_background = bool(self.background_data) and (mode == 'training')

        # These scalar tensors are passed *into* the compiled _process_one
        # function so TF doesn't need to retrace for different call-site values.
        bg_tensor_t    = self._background_tensor
        bg_frequency_t = tf.constant(float(background_frequency),    tf.float32)
        bg_vol_range_t = tf.constant(float(background_volume_range),  tf.float32)
        time_shift_t   = tf.constant(int(time_shift),                 tf.int32)
        use_bg_t       = tf.constant(use_background,                  tf.bool)

        process_one = self._process_one   # pre-compiled, do NOT redefine here

        def _map_fn(wav_path, label, silence_flag):
            # Thin wrapper: forwards the per-call scalars that are constant
            # within this dataset but can't be put in input_signature
            # because they differ between training/validation calls.
            return process_one(
                wav_path, label, silence_flag,
                bg_tensor_t, bg_frequency_t, bg_vol_range_t,
                time_shift_t, use_bg_t)

        ds = tf.data.Dataset.from_tensor_slices((
            tf.constant(files),
            tf.constant(word_indices,  dtype=tf.int32),
            tf.constant(silence_flags, dtype=tf.int32),
        ))
        ds = ds.map(_map_fn,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=(mode != 'training'))
        ds = ds.batch(sample_count).prefetch(1)

        for fp_batch, lbl_batch in ds:
            return fp_batch.numpy(), lbl_batch.numpy()

    # ------------------------------------------------------------------

    def get_features_for_wav(self, wav_filename, model_settings, sess=None):
        """Extract features from a single WAV file.  `sess` is ignored."""
        desired_samples = self._desired_samples
        raw     = io_ops.read_file(wav_filename)
        decoded = tf.audio.decode_wav(
            raw, desired_channels=1, desired_samples=desired_samples)
        features = self._process_audio(
            decoded.audio,
            tf.zeros([desired_samples, 1], tf.float32),
            tf.constant(0.0, tf.float32),
            tf.constant(1.0, tf.float32),
            tf.constant([[0, 0], [0, 0]], tf.int32),
            tf.constant([0, 0], tf.int32),
        )
        return features.numpy()

    # ------------------------------------------------------------------

    def get_unprocessed_data(self, how_many, model_settings, mode):
        """Return raw (unaugmented) audio arrays.  No `sess` needed."""
        candidates      = self.data_index[mode]
        desired_samples = self._desired_samples
        sample_count    = len(candidates) if how_many == -1 else how_many

        if mode == 'training' and how_many != -1:
            indices = np.random.randint(
                0, len(candidates), size=sample_count).tolist()
        else:
            indices = list(range(sample_count))

        files         = [candidates[i]['file'] for i in indices]
        silence_flags = [1 if candidates[i]['label'] == SILENCE_LABEL else 0
                         for i in indices]
        label_indices = [self.word_to_index[candidates[i]['label']]
                         for i in indices]

        load_one = self._load_one   # pre-compiled, do NOT redefine

        ds = tf.data.Dataset.from_tensor_slices((
            tf.constant(files),
            tf.constant(silence_flags, dtype=tf.int32),
        ))
        ds = ds.map(load_one, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(sample_count).prefetch(1)

        for audio_batch in ds:
            data   = audio_batch.numpy()
            labels = [self.words_list[li] for li in label_indices]
            return data, labels