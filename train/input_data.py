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

Refactored to TensorFlow v2 eager execution. All public method signatures are
preserved exactly — including the `sess` argument on get_data(),
get_features_for_wav(), etc. — so that call-sites require no changes.
`sess` is accepted but never used; pass None freely.
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

# Optional micro-frontend op.
try:
    from tensorflow.lite.experimental.microfrontend.python.ops import (
        audio_microfrontend_op as frontend_op,
    )
except ImportError:
    frontend_op = None

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


# ---------------------------------------------------------------------------
# Module-level helpers (unchanged signatures)
# ---------------------------------------------------------------------------

def prepare_words_list(wanted_words):
    """Prepends common tokens to the custom word list.

    Args:
      wanted_words: List of strings containing the custom words.

    Returns:
      List with the standard silence and unknown tokens added.
    """
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
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
    """Loads an audio file and returns a float PCM-encoded array of samples.

    Args:
      filename: Path to the .wav file to load.

    Returns:
      Numpy array holding the sample data as floats between -1.0 and 1.0.
    """
    wav_loader = io_ops.read_file(filename)
    wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
    return wav_decoder.audio.numpy().flatten()


def save_wav_file(filename, wav_data, sample_rate):
    """Saves audio sample data to a .wav audio file.

    Args:
      filename: Path to save the file to.
      wav_data: 2D array of float PCM-encoded audio data.
      sample_rate: Samples per second to encode in the file.
    """
    wav_data_tensor = tf.constant(np.reshape(wav_data, (-1, 1)), dtype=tf.float32)
    wav_encoder = tf.audio.encode_wav(wav_data_tensor, sample_rate)
    io_ops.write_file(filename, wav_encoder)


def get_features_range(model_settings):
    """Returns the expected min/max for generated features.

    Args:
      model_settings: Information about the current model being trained.

    Returns:
      Min/max float pair holding the range of features.

    Raises:
      Exception: If preprocessing mode isn't recognized.
    """
    if model_settings['preprocess'] == 'average':
        return 0.0, 127.5
    elif model_settings['preprocess'] == 'mfcc':
        return -247.0, 30.0
    elif model_settings['preprocess'] == 'micro':
        return 0.0, 26.0
    raise Exception(
        'Unknown preprocess mode "%s" (should be "mfcc", "average", or "micro")'
        % model_settings['preprocess']
    )


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
    # Download / extraction
    # ------------------------------------------------------------------

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        """Download and extract data set tar file.

        Args:
          data_url: Web location of the tar file containing the data set.
          dest_directory: File path to extract data to.
        """
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
                    'Failed to download URL: %s to folder: %s. Please make sure '
                    'you have enough free space and an internet connection.',
                    data_url, filepath)
                raise
            print()
            statinfo = os.stat(filepath)
            tf.get_logger().info(
                'Successfully downloaded %s (%d bytes).',
                filename, statinfo.st_size)
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    # ------------------------------------------------------------------
    # Data index
    # ------------------------------------------------------------------

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                            wanted_words, validation_percentage,
                            testing_percentage):
        """Prepares a list of the samples organized by set and label.

        Args:
          silence_percentage: How much of the resulting data should be silence.
          unknown_percentage: How much should be audio outside wanted classes.
          wanted_words: Labels of the classes we want to recognize.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.

        Raises:
          Exception: If expected files are not found.
        """
        random.seed(RANDOM_SEED)
        wanted_words_index = {
            word: idx + 2 for idx, word in enumerate(wanted_words)
        }
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}

        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(
                wav_path, validation_percentage, testing_percentage)
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
            set_size = len(self.data_index[set_index])
            silence_size = int(
                math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append(
                    {'label': SILENCE_LABEL, 'file': silence_wav_path})
            random.shuffle(unknown_index[set_index])
            unknown_size = int(
                math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(
                unknown_index[set_index][:unknown_size])

        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])

        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    # ------------------------------------------------------------------
    # Background data
    # ------------------------------------------------------------------

    def prepare_background_data(self):
        """Searches a folder for background noise audio and loads it into memory.

        Raises:
          Exception: If the background folder exists but is empty.
        """
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not gfile.Exists(background_dir):
            return

        search_path = os.path.join(
            self.data_dir, BACKGROUND_NOISE_DIR_NAME, '*.wav')
        for wav_path in gfile.Glob(search_path):
            wav_loader = io_ops.read_file(wav_path)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
            self.background_data.append(
                wav_decoder.audio.numpy().flatten())

        if not self.background_data:
            raise Exception(
                'No background wav files were found in ' + search_path)

    # ------------------------------------------------------------------
    # Processing graph (now a @tf.function for efficient repeated use)
    # ------------------------------------------------------------------

    def prepare_processing_graph(self, model_settings, summaries_dir):
        """Stores model settings and sets up the TensorBoard summary writer.

        In TF2 the per-sample processing is done eagerly in get_data() using
        _process_sample(), which is compiled via @tf.function for speed.

        Args:
          model_settings: Information about the current model being trained.
          summaries_dir: Path to save training summary information to.
        """
        self._model_settings = model_settings

        # Summary writer for data-pipeline images (optional).
        self.summary_writer_ = None
        if summaries_dir:
            self.summary_writer_ = tf.summary.create_file_writer(
                os.path.join(summaries_dir, 'data'))

        # Build a tf.function that processes a single sample tensor so the
        # graph is compiled once and reused across calls.
        preprocess = model_settings['preprocess']
        desired_samples = model_settings['desired_samples']
        window_size = model_settings['window_size_samples']
        window_stride = model_settings['window_stride_samples']
        fingerprint_width = model_settings['fingerprint_width']
        average_window_width = model_settings.get('average_window_width', -1)
        sample_rate = model_settings['sample_rate']

        @tf.function
        def _process_audio(foreground, background_data,
                            background_volume, foreground_volume,
                            time_shift_padding, time_shift_offset):
            """Applies augmentation + feature extraction to one audio clip.

            Args:
              foreground:          [desired_samples, 1] float32 audio.
              background_data:     [desired_samples, 1] float32 noise.
              background_volume:   Scalar float — noise gain.
              foreground_volume:   Scalar float — speech gain (0 = silence).
              time_shift_padding:  [2, 2] int32 pad amounts.
              time_shift_offset:   [2]    int32 slice start.

            Returns:
              features: Flattened 1-D float32 feature vector.
            """
            scaled = tf.multiply(foreground, foreground_volume)
            padded = tf.pad(scaled, time_shift_padding, mode='CONSTANT')
            sliced = tf.slice(padded, time_shift_offset, [desired_samples, -1])
            bg_mul = tf.multiply(background_data, background_volume)
            mixed = tf.clip_by_value(tf.add(bg_mul, sliced), -1.0, 1.0)

            spectrogram = audio_ops.audio_spectrogram(
                mixed,
                window_size=window_size,
                stride=window_stride,
                magnitude_squared=True)

            if preprocess == 'average':
                output = tf.nn.pool(
                    input=tf.expand_dims(spectrogram, -1),
                    window_shape=[1, average_window_width],
                    strides=[1, average_window_width],
                    pooling_type='AVG',
                    padding='SAME')
            elif preprocess == 'mfcc':
                # sample_rate must be a Python int for audio_ops.mfcc
                output = audio_ops.mfcc(
                    spectrogram,
                    sample_rate,
                    dct_coefficient_count=fingerprint_width)
            elif preprocess == 'micro':
                if not frontend_op:
                    raise Exception(
                        'Micro frontend op is not available when running '
                        'TensorFlow directly from Python.')
                window_size_ms = (window_size * 1000) / sample_rate
                window_step_ms = (window_stride * 1000) / sample_rate
                int16_input = tf.cast(
                    tf.multiply(mixed, 32768), tf.int16)
                micro_frontend = frontend_op.audio_microfrontend(
                    int16_input,
                    sample_rate=sample_rate,
                    window_size=window_size_ms,
                    window_step=window_step_ms,
                    num_channels=fingerprint_width,
                    out_scale=1,
                    out_type=tf.float32)
                output = tf.multiply(micro_frontend, (10.0 / 256.0))
            else:
                raise ValueError(
                    'Unknown preprocess mode "%s"' % preprocess)

            return tf.reshape(output, [-1])

        self._process_audio = _process_audio

    # ------------------------------------------------------------------
    # Public interface (all sess arguments kept for API compatibility)
    # ------------------------------------------------------------------

    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.

        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.

        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def get_data(self, how_many, offset, model_settings, background_frequency,
                 background_volume_range, time_shift, mode, sess=None):
        """Gather samples from the data set, applying transformations as needed.

        Args:
          how_many: Desired number of samples to return. -1 means all.
          offset: Where to start when fetching deterministically.
          model_settings: Information about the current model being trained.
          background_frequency: How many clips will have background noise (0–1).
          background_volume_range: How loud the background noise will be.
          time_shift: How much to randomly shift the clips by in time.
          mode: Which partition, must be 'training', 'validation', or 'testing'.
          sess: Unused — kept for API compatibility with TF1 call-sites.

        Returns:
          Tuple (data, labels):
            data:   np.ndarray [sample_count, fingerprint_size] float32.
            labels: np.ndarray [sample_count] int.

        Raises:
          ValueError: If background samples are too short.
        """
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        data = np.zeros((sample_count, model_settings['fingerprint_size']),
                        dtype=np.float32)
        labels = np.zeros(sample_count, dtype=np.int32)

        desired_samples = model_settings['desired_samples']
        use_background = self.background_data and (mode == 'training')
        pick_deterministically = (mode != 'training')

        for i in range(offset, offset + sample_count):
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]

            # ---- Load foreground wav ----
            wav_loader = io_ops.read_file(sample['file'])
            wav_decoder = tf.audio.decode_wav(
                wav_loader, desired_channels=1,
                desired_samples=desired_samples)
            foreground = wav_decoder.audio  # [desired_samples, 1]

            # ---- Time shift ----
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]

            # ---- Background noise ----
            if use_background or sample['label'] == SILENCE_LABEL:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                if len(background_samples) <= desired_samples:
                    raise ValueError(
                        'Background sample is too short! Need more than %d '
                        'samples but only %d were found' %
                        (desired_samples, len(background_samples)))
                background_offset = np.random.randint(
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
                    background_volume = 0.0
            else:
                background_reshaped = np.zeros(
                    [desired_samples, 1], dtype=np.float32)
                background_volume = 0.0

            foreground_volume = (
                0.0 if sample['label'] == SILENCE_LABEL else 1.0)

            # ---- Feature extraction ----
            features = self._process_audio(
                foreground,
                tf.constant(background_reshaped, dtype=tf.float32),
                tf.constant(background_volume, dtype=tf.float32),
                tf.constant(foreground_volume, dtype=tf.float32),
                tf.constant(time_shift_padding, dtype=tf.int32),
                tf.constant(time_shift_offset, dtype=tf.int32),
            )

            data[i - offset, :] = features.numpy()
            labels[i - offset] = self.word_to_index[sample['label']]

            # Optional TensorBoard image summary (best-effort).
            if self.summary_writer_:
                with self.summary_writer_.as_default():
                    tf.summary.image(
                        'features',
                        tf.reshape(features, [1, 1, -1, 1]),
                        step=i)

        return data, labels

    def get_features_for_wav(self, wav_filename, model_settings, sess=None):
        """Applies the feature transformation process to the input wav.

        Args:
          wav_filename: The path to the input audio file.
          model_settings: Information about the current model being trained.
          sess: Unused — kept for API compatibility with TF1 call-sites.

        Returns:
          Numpy array containing the generated features.
        """
        desired_samples = model_settings['desired_samples']
        wav_loader = io_ops.read_file(wav_filename)
        wav_decoder = tf.audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=desired_samples)

        features = self._process_audio(
            wav_decoder.audio,
            tf.zeros([desired_samples, 1], dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(1.0, dtype=tf.float32),
            tf.constant([[0, 0], [0, 0]], dtype=tf.int32),
            tf.constant([0, 0], dtype=tf.int32),
        )
        return features.numpy()

    def get_unprocessed_data(self, how_many, model_settings, mode):
        """Retrieve raw sample data with no transformations.

        Args:
          how_many: Desired number of samples to return. -1 means all.
          model_settings: Information about the current model being trained.
          mode: Which partition, must be 'training', 'validation', or 'testing'.

        Returns:
          Tuple (data, labels):
            data:   np.ndarray [sample_count, desired_samples] float32.
            labels: List of word-string labels.
        """
        candidates = self.data_index[mode]
        sample_count = len(candidates) if how_many == -1 else how_many
        desired_samples = model_settings['desired_samples']

        data = np.zeros((sample_count, desired_samples), dtype=np.float32)
        labels = []

        for i in range(sample_count):
            if how_many == -1:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]

            wav_loader = io_ops.read_file(sample['file'])
            wav_decoder = tf.audio.decode_wav(
                wav_loader, desired_channels=1,
                desired_samples=desired_samples)

            foreground_volume = (
                0.0 if sample['label'] == SILENCE_LABEL else 1.0)
            scaled = tf.multiply(wav_decoder.audio,
                                 tf.constant(foreground_volume, tf.float32))
            data[i, :] = scaled.numpy().flatten()

            label_index = self.word_to_index[sample['label']]
            labels.append(self.words_list[label_index])

        return data, labels