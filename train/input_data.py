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
"""Audio data loading and preprocessing for keyword recognition.

TF1 deprecation changelog  (previous version -> this version)
--------------------------------------------------------------
REMOVED (TF1 training pipeline no longer supported):
  prepare_processing_graph   -- TF1 graph builder with placeholders
  get_data (sess arg)        -- sess.run feed_dict loop
  get_features_for_wav (sess)-- sess.run on a single file
  get_data_as_tf_dataset     -- TF1-session bridge (now fully eager)

REPLACED WITH TF2 eager equivalents:
  preprocess_audio_tf2()     -- module-level eager function for all three
                                modes.  mfcc and average use tf.signal;
                                micro uses a contained TF1 block (see note).
  AudioProcessor.get_data()  -- pure numpy/eager, no session argument
  AudioProcessor.get_features_for_wav() -- eager, no session argument
  AudioProcessor.get_data_as_tf_dataset() -- wraps the above, no session

ALREADY TF2 (unchanged):
  load_wav_file, save_wav_file, prepare_words_list, which_set,
  get_features_range, maybe_download_and_extract_dataset,
  prepare_data_index, prepare_background_data

micro preprocessing -- why it uses a separate path
----------------------------------------------------
The audio_microfrontend C++ op has no tf.signal equivalent and requires
the tensorflow.lite.experimental.microfrontend package (available in
some pip wheels and all Bazel builds).

  * For NEW training  -- use preprocess='mfcc' or preprocess='average'.
    Both are fully eager (tf.signal) with no extra dependencies.

  * For inference on an EXISTING micro model -- _preprocess_micro_eager()
    calls the op directly as an EagerTensor operation (no TF1 session or
    placeholder needed).  It is isolated so the rest of this file has no
    dependency on the microfrontend package.

  * The op is called with all parameters matching the reference file
    (tensorflow.lite.experimental.microfrontend.ops.audio_microfrontend_op)
    exactly, including PCAN, log scaling, and all band/smoothing defaults.
"""
import hashlib
import math
import os
import os.path
import random
import re
import sys
import tarfile
import urllib.request

import numpy as np
import tensorflow as tf

# microfrontend op -- only available in Bazel builds.
try:
    from tensorflow.lite.experimental.microfrontend.python.ops \
        import audio_microfrontend_op as frontend_op
    _MICRO_FRONTEND_AVAILABLE = True
except ImportError:
    frontend_op = None
    _MICRO_FRONTEND_AVAILABLE = False

MAX_NUM_WAVS_PER_CLASS    = 2 ** 27 - 1   # ~134 M
SILENCE_LABEL             = '_silence_'
SILENCE_INDEX             = 0
UNKNOWN_WORD_LABEL        = '_unknown_'
UNKNOWN_WORD_INDEX        = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED               = 59185


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def prepare_words_list(wanted_words):
    """Prepends standard tokens to the custom word list."""
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    """Assigns a file to a partition deterministically via SHA-1 hash."""
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hex_hash  = hashlib.sha1(hash_name.encode()).hexdigest()
    pct_hash  = ((int(hex_hash, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
                 * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if pct_hash < validation_percentage:
        return 'validation'
    if pct_hash < (testing_percentage + validation_percentage):
        return 'testing'
    return 'training'


def load_wav_file(filename):
    """Loads a WAV file and returns a float32 PCM array."""
    raw      = tf.io.read_file(filename)
    audio, _ = tf.audio.decode_wav(raw, desired_channels=1)
    return audio.numpy().flatten()


def save_wav_file(filename, wav_data, sample_rate):
    """Saves float32 PCM data as a WAV file."""
    wav_tensor = tf.constant(np.reshape(wav_data, (-1, 1)), dtype=tf.float32)
    encoded    = tf.audio.encode_wav(wav_tensor, sample_rate=int(sample_rate))
    tf.io.write_file(filename, encoded)


def get_features_range(model_settings):
    """Returns the (min, max) value range for fingerprints in the given mode."""
    mode = model_settings['preprocess']
    if mode == 'average': return 0.0,    127.5
    if mode == 'mfcc':    return -247.0, 30.0
    if mode == 'micro':   return 0.0,    26.0
    raise Exception('Unknown preprocess mode "%s"' % mode)


# ---------------------------------------------------------------------------
# Eager feature extraction
# ---------------------------------------------------------------------------

def _preprocess_micro_eager(waveform, model_settings):
    """Applies the audio_microfrontend op eagerly to a full 1-second waveform.

    Uses the same op as the TFLM reference file
    (tensorflow.lite.experimental.microfrontend.ops.audio_microfrontend_op)
    but called directly as an EagerTensor operation — no TF1 session or
    placeholder required.

    All parameters match the reference file's audio_microfrontend() signature
    exactly.  out_type=tf.float32 is used (instead of the reference default
    uint16) so the output is already floating-point, consistent with the
    original train.py pipeline.  The 10.0/256.0 scale factor is the same
    scaling applied by the original TF1 training code.

    Args:
        waveform:       1-D float32 numpy array, values in [-1.0, 1.0],
                        length == model_settings['desired_samples'].
        model_settings: Dict from model_settings.prepare_model_settings().

    Returns:
        2-D float32 numpy array [spectrogram_length, fingerprint_width].

    Raises:
        RuntimeError: If the microfrontend .so is not available.
    """
    if not _MICRO_FRONTEND_AVAILABLE:
        raise RuntimeError(
            'preprocess=micro requires the audio_microfrontend C++ op '
            '(tensorflow.lite.experimental.microfrontend), available in '
            'Bazel builds and some pip wheels.\n'
            'Use preprocess=mfcc or preprocess=average for a fully TF2 '
            'eager pipeline without this dependency.')

    sample_rate    = model_settings['sample_rate']
    window_size_ms = model_settings['window_size_samples'] * 1000 / sample_rate
    window_step_ms = model_settings['window_stride_samples'] * 1000 / sample_rate

    # Convert float32 PCM [-1, 1] to int16 — matches the reference file:
    #   int16_input = tf.cast(tf.multiply(background_clamp, 32768), tf.int16)
    int16  = tf.cast(tf.multiply(tf.constant(waveform), 32768), tf.int16)

    # Call the op eagerly — no session needed.
    # Parameters match the reference audio_microfrontend() signature exactly.
    mfe = frontend_op.audio_microfrontend(
        int16,
        sample_rate=sample_rate,
        window_size=window_size_ms,
        window_step=window_step_ms,
        num_channels=model_settings['fingerprint_width'],
        upper_band_limit=7500.0,
        lower_band_limit=125.0,
        smoothing_bits=10,
        even_smoothing=0.025,
        odd_smoothing=0.06,
        min_signal_remaining=0.05,
        enable_pcan=True,
        pcan_strength=0.95,
        pcan_offset=80.0,
        gain_bits=21,
        enable_log=True,
        scale_shift=6,
        left_context=0,
        right_context=0,
        frame_stride=1,
        zero_padding=False,
        out_scale=1,
        out_type=tf.float32)

    return tf.multiply(mfe, 10.0 / 256.0).numpy()  # [spectrogram_length, fingerprint_width]


def preprocess_audio_tf2(waveform, model_settings):
    """Converts a raw waveform into a flattened fingerprint vector.

    Fully eager for mfcc and average.  micro delegates to
    _preprocess_micro_tf1 (isolated TF1 block).

    mfcc path:
        tf.signal.stft -> mel filterbank -> log -> DCT (type-II)
        NOTE: Values differ numerically from gen_audio_ops.mfcc.
        Use consistent preprocessing end-to-end.

    average path:
        tf.signal.stft -> magnitude -> average pool over frequency bins.

    Args:
        waveform:       1-D float32 numpy array, length == desired_samples.
        model_settings: Dict from model_settings.prepare_model_settings().

    Returns:
        1-D float32 numpy array of length fingerprint_size.
    """
    preprocess = model_settings['preprocess']

    if preprocess == 'micro':
        return _preprocess_micro_eager(waveform, model_settings).flatten().astype(np.float32)

    wav_tensor = tf.constant(waveform, dtype=tf.float32)
    stft       = tf.signal.stft(
        wav_tensor,
        frame_length=model_settings['window_size_samples'],
        frame_step=model_settings['window_stride_samples'],
        pad_end=False)
    magnitude  = tf.abs(stft)   # [T, F_bins]

    if preprocess == 'mfcc':
        sample_rate  = model_settings['sample_rate']
        num_mel_bins = 128
        num_mfccs    = model_settings['fingerprint_width']

        linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=magnitude.shape[-1],
            sample_rate=sample_rate,
            lower_edge_hertz=20.0,
            upper_edge_hertz=sample_rate / 2.0)

        mel_spec = tf.tensordot(magnitude, linear_to_mel, 1)
        log_mel  = tf.math.log(mel_spec + 1e-6)
        mfccs    = tf.signal.dcts(log_mel, n=num_mfccs, type=2,
                                  axis=-1)[..., :num_mfccs]
        return mfccs.numpy().flatten().astype(np.float32)

    if preprocess == 'average':
        avg_width = model_settings['average_window_width']
        expanded  = tf.reshape(magnitude,
                               [tf.shape(magnitude)[0], 1,
                                tf.shape(magnitude)[1], 1])
        pooled    = tf.nn.avg_pool2d(
            expanded,
            ksize=[1, 1, avg_width, 1],
            strides=[1, 1, avg_width, 1],
            padding='SAME')
        result    = tf.squeeze(pooled, axis=[1, 3])
        return result.numpy().flatten().astype(np.float32)

    raise ValueError('Unknown preprocess mode "%s"' % preprocess)


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------

class AudioProcessor:
    """Handles loading, partitioning, and preparing audio training data.

    Public API is fully TF2 eager -- no session argument on any method.
    """

    def __init__(self, data_url, data_dir, silence_percentage,
                 unknown_percentage, wanted_words, validation_percentage,
                 testing_percentage, model_settings, summaries_dir):
        self.model_settings = model_settings
        if data_dir:
            self.data_dir = data_dir
            self.maybe_download_and_extract_dataset(data_url, data_dir)
            self.prepare_data_index(silence_percentage, unknown_percentage,
                                    wanted_words, validation_percentage,
                                    testing_percentage)
            self.prepare_background_data()

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        """Downloads and extracts the dataset tarball if not already present."""
        if not data_url:
            return
        os.makedirs(dest_directory, exist_ok=True)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not tf.io.gfile.exists(filepath):
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
                    'Failed to download URL: %s to folder: %s.',
                    data_url, filepath)
                raise
            print()
            tf.get_logger().info('Successfully downloaded %s (%d bytes)',
                                 filename, os.stat(filepath).st_size)
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                           wanted_words, validation_percentage,
                           testing_percentage):
        """Builds the per-partition list of (file, label) dicts."""
        random.seed(RANDOM_SEED)
        wanted_words_index = {w: i + 2 for i, w in enumerate(wanted_words)}
        self.data_index    = {'validation': [], 'testing': [], 'training': []}
        unknown_index      = {'validation': [], 'testing': [], 'training': []}
        all_words          = {}

        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in tf.io.gfile.glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
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
        for word in wanted_words:
            if word not in all_words:
                raise Exception(
                    'Expected to find ' + word + ' in labels but only found ' +
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
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_background_data(self):
        """Loads _background_noise_ WAV files into self.background_data."""
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not tf.io.gfile.exists(background_dir):
            return
        for wav_path in tf.io.gfile.glob(os.path.join(background_dir, '*.wav')):
            raw      = tf.io.read_file(wav_path)
            audio, _ = tf.audio.decode_wav(raw, desired_channels=1)
            self.background_data.append(audio.numpy().flatten())
        if not self.background_data:
            raise Exception('No background wav files found in ' + background_dir)

    def set_size(self, mode):
        """Returns the number of samples in the given partition."""
        return len(self.data_index[mode])

    def get_data(self, how_many, offset, model_settings,
                 background_frequency, background_volume_range,
                 time_shift, mode):
        """Gathers samples applying distortions; fully TF2 eager, no session arg.

        For training, samples are drawn randomly.  For validation/testing they
        are deterministic so metrics are reproducible.

        Args:
            how_many:                 Sample count; -1 returns all.
            offset:                   Start index for deterministic fetch.
            model_settings:           Dict from model_settings.prepare_model_settings().
            background_frequency:     Fraction of training clips to add noise.
            background_volume_range:  Max background noise volume (0-1).
            time_shift:               Max time-shift in samples.
            mode:                     'training', 'validation', or 'testing'.

        Returns:
            (data float32 [N, fingerprint_size], labels int64 [N])
        """
        candidates       = self.data_index[mode]
        sample_count     = (len(candidates) if how_many == -1
                            else max(0, min(how_many, len(candidates) - offset)))
        fingerprint_size = model_settings['fingerprint_size']
        desired_samples  = model_settings['desired_samples']
        data             = np.zeros((sample_count, fingerprint_size), dtype=np.float32)
        labels           = np.zeros(sample_count, dtype=np.int64)

        use_background         = bool(self.background_data) and (mode == 'training')
        pick_deterministically = (mode != 'training')

        for i in range(sample_count):
            sample_index = (i + offset) if pick_deterministically else np.random.randint(len(candidates))
            sample       = candidates[sample_index]

            raw          = tf.io.read_file(sample['file'])
            audio, _     = tf.audio.decode_wav(raw, desired_channels=1,
                                               desired_samples=desired_samples)
            waveform     = audio.numpy().flatten()

            # Time-shift augmentation
            if time_shift > 0:
                shift = np.random.randint(-time_shift, time_shift)
                if shift > 0:
                    waveform = np.pad(waveform, (shift, 0))[:desired_samples]
                elif shift < 0:
                    waveform = np.pad(waveform, (0, -shift))[-desired_samples:]

            # Foreground scaling
            waveform = waveform * (0.0 if sample['label'] == SILENCE_LABEL else 1.0)

            # Background noise
            if use_background or sample['label'] == SILENCE_LABEL:
                bg_idx     = np.random.randint(len(self.background_data))
                bg_samples = self.background_data[bg_idx]
                if len(bg_samples) <= desired_samples:
                    raise ValueError(
                        'Background sample too short: need > %d samples but got %d'
                        % (desired_samples, len(bg_samples)))
                bg_offset = np.random.randint(0, len(bg_samples) - desired_samples)
                bg_clip   = bg_samples[bg_offset: bg_offset + desired_samples]

                if sample['label'] == SILENCE_LABEL:
                    bg_volume = np.random.uniform(0, 1)
                elif np.random.uniform(0, 1) < background_frequency:
                    bg_volume = np.random.uniform(0, background_volume_range)
                else:
                    bg_volume = 0.0
                waveform = np.clip(waveform + bg_clip * bg_volume, -1.0, 1.0)

            data[i]   = preprocess_audio_tf2(waveform, model_settings)
            labels[i] = self.word_to_index[sample['label']]

        return data, labels

    def get_features_for_wav(self, wav_filename, model_settings):
        """Returns a fingerprint vector for a single WAV file.  No session arg.

        Args:
            wav_filename:   Path to .wav file.
            model_settings: Dict from model_settings.prepare_model_settings().

        Returns:
            1-D float32 numpy array of length fingerprint_size.
        """
        desired_samples = model_settings['desired_samples']
        raw             = tf.io.read_file(wav_filename)
        audio, _        = tf.audio.decode_wav(raw, desired_channels=1,
                                              desired_samples=desired_samples)
        return preprocess_audio_tf2(audio.numpy().flatten(), model_settings)

    def get_unprocessed_data(self, how_many, model_settings, mode):
        """Returns raw waveforms (no preprocessing) for debugging.

        Args:
            how_many:       Sample count; -1 returns all.
            model_settings: Dict from model_settings.prepare_model_settings().
            mode:           'training', 'validation', or 'testing'.

        Returns:
            (waveforms float32 [N, desired_samples], labels int64 [N])
        """
        candidates      = self.data_index[mode]
        sample_count    = (len(candidates) if how_many == -1
                           else min(how_many, len(candidates)))
        desired_samples = model_settings['desired_samples']
        waveforms = np.zeros((sample_count, desired_samples), dtype=np.float32)
        labels    = np.zeros(sample_count, dtype=np.int64)
        for i, sample in enumerate(candidates[:sample_count]):
            raw          = tf.io.read_file(sample['file'])
            audio, _     = tf.audio.decode_wav(raw, desired_channels=1,
                                               desired_samples=desired_samples)
            waveforms[i] = audio.numpy().flatten()
            labels[i]    = self.word_to_index[sample['label']]
        return waveforms, labels

    def get_data_as_tf_dataset(self, model_settings, mode, batch_size=64,
                                background_frequency=0.0,
                                background_volume_range=0.1,
                                time_shift=0, shuffle=False):
        """Returns a batched tf.data.Dataset; fully eager, no session.

        Args:
            model_settings:          Dict from model_settings.prepare_model_settings().
            mode:                    'training', 'validation', or 'testing'.
            batch_size:              Samples per batch.
            background_frequency:    Noise augmentation fraction.
            background_volume_range: Max noise volume.
            time_shift:              Max time-shift in samples.
            shuffle:                 Whether to shuffle before batching.

        Returns:
            tf.data.Dataset of (fingerprints float32 [B,fingerprint_size],
                                labels       int32   [B]).
        """
        fingerprints, labels = self.get_data(
            how_many=-1, offset=0,
            model_settings=model_settings,
            background_frequency=background_frequency,
            background_volume_range=background_volume_range,
            time_shift=time_shift,
            mode=mode)
        ds = tf.data.Dataset.from_tensor_slices(
            (fingerprints.astype(np.float32), labels.astype(np.int32)))
        if shuffle:
            ds = ds.shuffle(len(fingerprints), reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)