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
"""Model definitions for simple speech recognition — TF2 refactor."""

import hashlib
import math
import os
import random
import re
import sys
import tarfile
import urllib

import numpy as np
import tensorflow as tf

# TF2: use public API instead of internal gen_audio_ops
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

# TF2: eager execution is ON by default — no disable needed

try:
    from tensorflow.lite.experimental.microfrontend.python.ops import (
        audio_microfrontend_op as frontend_op,
    )
except ImportError:
    frontend_op = None

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = "_silence_"
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = "_unknown_"
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = "_background_noise_"
RANDOM_SEED = 59185


# ---------------------------------------------------------------------------
# Standalone helpers
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

    Uses a stable hash of the filename so partitions remain consistent
    even when new files are added later.

    Args:
        filename: File path of the data sample.
        validation_percentage: How much of the data set to use for validation.
        testing_percentage: How much of the data set to use for testing.

    Returns:
        String — one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = (
        (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
        * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    )
    if percentage_hash < validation_percentage:
        return "validation"
    elif percentage_hash < (testing_percentage + validation_percentage):
        return "testing"
    return "training"


def load_wav_file(filename):
    """Loads an audio file and returns a float PCM-encoded array of samples.

    TF2: reads the file with tf.io and decodes with tf.audio eagerly.

    Args:
        filename: Path to the .wav file to load.

    Returns:
        Numpy array of float samples in [-1.0, 1.0].
    """
    raw = tf.io.read_file(filename)
    audio, _ = tf.audio.decode_wav(raw, desired_channels=1)
    return audio.numpy().flatten()


def save_wav_file(filename, wav_data, sample_rate):
    """Saves audio sample data to a .wav audio file.

    TF2: encodes eagerly with tf.audio and writes with tf.io.

    Args:
        filename: Path to save the file to.
        wav_data: 2-D array of float PCM-encoded audio data.
        sample_rate: Samples per second to encode in the file.
    """
    reshaped = np.reshape(wav_data, (-1, 1)).astype(np.float32)
    encoded = tf.audio.encode_wav(reshaped, sample_rate)
    tf.io.write_file(filename, encoded)


def get_features_range(model_settings):
    """Returns the expected min/max for generated features.

    Args:
        model_settings: Information about the current model being trained.

    Returns:
        (min, max) float pair for the feature range.

    Raises:
        Exception: If preprocessing mode isn't recognised.
    """
    mode = model_settings["preprocess"]
    if mode == "average":
        return 0.0, 127.5
    elif mode == "mfcc":
        return -247.0, 30.0
    elif mode == "micro":
        return 0.0, 26.0
    raise Exception(
        f'Unknown preprocess mode "{mode}" (should be "mfcc", "average", or "micro")'
    )


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------

class AudioProcessor:
    """Handles loading, partitioning, and preparing audio training data.

    TF2 notes
    ---------
    * All ``sess`` / ``sess.run()`` calls have been removed.
    * The processing graph is replaced by a ``tf.function``-wrapped
      ``_run_processing_graph`` method that executes eagerly on first call
      and is traced/compiled on subsequent calls for speed.
    * ``get_data`` no longer accepts a ``sess`` argument.
    * ``prepare_processing_graph`` builds concrete ``tf.Tensor`` objects
      (no placeholders); inputs are fed as arguments to the tf.function.
    * Summary writing is done via ``tf.summary`` v2 API.
    """

    def __init__(
        self,
        data_url,
        data_dir,
        silence_percentage,
        unknown_percentage,
        wanted_words,
        validation_percentage,
        testing_percentage,
        model_settings,
        summaries_dir=None,
    ):
        if data_dir:
            self.data_dir = data_dir
            self.maybe_download_and_extract_dataset(data_url, data_dir)
            self.prepare_data_index(
                silence_percentage,
                unknown_percentage,
                wanted_words,
                validation_percentage,
                testing_percentage,
            )
            self.prepare_background_data()
        self.prepare_processing_graph(model_settings, summaries_dir)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        """Download and extract dataset tar file if not already present."""
        if not data_url:
            return
        os.makedirs(dest_directory, exist_ok=True)
        filename = data_url.split("/")[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    "\r>> Downloading %s %.1f%%"
                    % (filename, float(count * block_size) / float(total_size) * 100.0)
                )
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            except Exception:
                tf.get_logger().error(
                    "Failed to download URL: %s to folder: %s. "
                    "Please make sure you have enough free space and an internet connection.",
                    data_url,
                    filepath,
                )
                raise
            print()
            statinfo = os.stat(filepath)
            tf.get_logger().info(
                "Successfully downloaded %s (%d bytes).", filename, statinfo.st_size
            )
            tarfile.open(filepath, "r:gz").extractall(dest_directory)

    def prepare_data_index(
        self,
        silence_percentage,
        unknown_percentage,
        wanted_words,
        validation_percentage,
        testing_percentage,
    ):
        """Prepares a list of samples organised by partition and label."""
        random.seed(RANDOM_SEED)
        wanted_words_index = {w: i + 2 for i, w in enumerate(wanted_words)}
        self.data_index = {"validation": [], "testing": [], "training": []}
        unknown_index = {"validation": [], "testing": [], "training": []}
        all_words = {}

        search_path = os.path.join(self.data_dir, "*", "*.wav")
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            if word in wanted_words_index:
                self.data_index[set_index].append({"label": word, "file": wav_path})
            else:
                unknown_index[set_index].append({"label": word, "file": wav_path})

        if not all_words:
            raise Exception("No .wavs found at " + search_path)
        for wanted_word in wanted_words:
            if wanted_word not in all_words:
                raise Exception(
                    f"Expected to find {wanted_word} in labels but only found "
                    + ", ".join(all_words.keys())
                )

        silence_wav_path = self.data_index["training"][0]["file"]
        for set_index in ["validation", "testing", "training"]:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append(
                    {"label": SILENCE_LABEL, "file": silence_wav_path}
                )
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])

        for set_index in ["validation", "testing", "training"]:
            random.shuffle(self.data_index[set_index])

        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            self.word_to_index[word] = (
                wanted_words_index[word]
                if word in wanted_words_index
                else UNKNOWN_WORD_INDEX
            )
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_background_data(self):
        """Loads background noise wav files into memory (TF2 eager)."""
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not gfile.Exists(background_dir):
            return

        search_path = os.path.join(background_dir, "*.wav")
        for wav_path in gfile.Glob(search_path):
            # TF2: eager decode — no session needed
            raw = tf.io.read_file(wav_path)
            audio, _ = tf.audio.decode_wav(raw, desired_channels=1)
            self.background_data.append(audio.numpy().flatten())

        if not self.background_data:
            raise Exception("No background wav files were found in " + search_path)

    # ------------------------------------------------------------------
    # Processing graph (TF2 — tf.function replaces the static graph)
    # ------------------------------------------------------------------

    def prepare_processing_graph(self, model_settings, summaries_dir):
        """Stores model settings and sets up the TF2 summary writer.

        In TF2 the audio processing pipeline is implemented as a
        ``tf.function`` (``_process_sample``) rather than a static graph
        with placeholder tensors.  This gives the same performance benefit
        via tracing while remaining compatible with eager execution.

        Args:
            model_settings: Information about the current model being trained.
            summaries_dir: Path to save training summary information to.
        """
        self.model_settings_ = model_settings
        self.summary_writer_ = None
        if summaries_dir:
            self.summary_writer_ = tf.summary.create_file_writer(
                os.path.join(summaries_dir, "data")
            )

    @tf.function
    def _process_sample(
        self,
        wav_filename,
        time_shift_padding,
        time_shift_offset,
        background_data,
        background_volume,
        foreground_volume,
    ):
        """Applies audio augmentation and feature extraction to one sample.

        This method is traced by ``tf.function`` on first call, giving
        graph-mode performance while keeping readable eager-style code.

        Args:
            wav_filename:       Scalar string tensor — path to the WAV file.
            time_shift_padding: [2, 2] int32 tensor for tf.pad.
            time_shift_offset:  [2]    int32 tensor for tf.slice.
            background_data:    [desired_samples, 1] float32 tensor.
            background_volume:  Scalar float32.
            foreground_volume:  Scalar float32 (0 = silence, 1 = normal).

        Returns:
            Scalar summary string and the feature tensor (output_).
        """
        model_settings = self.model_settings_
        desired_samples = model_settings["desired_samples"]

        # --- load & decode WAV -------------------------------------------------
        raw = io_ops.read_file(wav_filename)
        wav_decoder = tf.audio.decode_wav(
            raw, desired_channels=1, desired_samples=desired_samples
        )

        # --- volume scaling ----------------------------------------------------
        scaled_foreground = tf.multiply(wav_decoder.audio, foreground_volume)

        # --- time shift --------------------------------------------------------
        padded = tf.pad(
            tensor=scaled_foreground,
            paddings=time_shift_padding,
            mode="CONSTANT",
        )
        sliced = tf.slice(padded, time_shift_offset, [desired_samples, -1])

        # --- background mixing -------------------------------------------------
        background_mul = tf.multiply(background_data, background_volume)
        background_add = tf.add(background_mul, sliced)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

        # --- feature extraction ------------------------------------------------
        spectrogram = audio_ops.audio_spectrogram(
            background_clamp,
            window_size=model_settings["window_size_samples"],
            stride=model_settings["window_stride_samples"],
            magnitude_squared=True,
        )

        preprocess = model_settings["preprocess"]
        if preprocess == "average":
            output_ = tf.nn.pool(
                input=tf.expand_dims(spectrogram, -1),
                window_shape=[1, model_settings["average_window_width"]],
                strides=[1, model_settings["average_window_width"]],
                pooling_type="AVG",
                padding="SAME",
            )
        elif preprocess == "mfcc":
            output_ = audio_ops.mfcc(
                spectrogram,
                wav_decoder.sample_rate,
                dct_coefficient_count=model_settings["fingerprint_width"],
            )
        elif preprocess == "micro":
            if not frontend_op:
                raise Exception(
                    "Micro frontend op is not available when running TensorFlow "
                    "directly from Python; build and run through Bazel instead."
                )
            sample_rate = model_settings["sample_rate"]
            window_size_ms = (model_settings["window_size_samples"] * 1000) / sample_rate
            window_step_ms = (model_settings["window_stride_samples"] * 1000) / sample_rate
            int16_input = tf.cast(tf.multiply(background_clamp, 32768), tf.int16)
            micro_frontend = frontend_op.audio_microfrontend(
                int16_input,
                sample_rate=sample_rate,
                window_size=window_size_ms,
                window_step=window_step_ms,
                num_channels=model_settings["fingerprint_width"],
                out_scale=1,
                out_type=tf.float32,
            )
            output_ = tf.multiply(micro_frontend, (10.0 / 256.0))
        else:
            raise ValueError(
                f'Unknown preprocess mode "{preprocess}" '
                '(should be "mfcc", "average", or "micro")'
            )

        return output_

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_size(self, mode):
        """Returns the number of samples in the given partition.

        Args:
            mode: 'training', 'validation', or 'testing'.

        Returns:
            Integer sample count.
        """
        return len(self.data_index[mode])

    def get_data(
        self,
        how_many,
        offset,
        model_settings,
        background_frequency,
        background_volume_range,
        time_shift,
        mode,
    ):
        """Gather samples from the dataset, applying augmentation transforms.

        TF2 version — ``sess`` parameter has been removed entirely.

        When mode is 'training' a random selection is returned; otherwise
        the first N clips in the partition are used so validation metrics
        are reproducible.

        Args:
            how_many:                Desired number of samples (-1 = all).
            offset:                  Start index for deterministic fetching.
            model_settings:          Model configuration dict.
            background_frequency:    Fraction of clips that get background noise.
            background_volume_range: Max background noise amplitude.
            time_shift:              Max random time-shift in samples.
            mode:                    'training', 'validation', or 'testing'.

        Returns:
            (data, labels) — numpy arrays of features and integer label indexes.

        Raises:
            ValueError: If background samples are too short.
        """
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        data = np.zeros((sample_count, model_settings["fingerprint_size"]))
        labels = np.zeros(sample_count)
        desired_samples = model_settings["desired_samples"]
        use_background = self.background_data and (mode == "training")
        pick_deterministically = mode != "training"

        step = 0  # tracks position inside the output arrays
        for i in range(offset, offset + sample_count):
            # --- pick sample ---------------------------------------------------
            sample_index = i if (how_many == -1 or pick_deterministically) \
                else np.random.randint(len(candidates))
            sample = candidates[sample_index]

            # --- time shift ----------------------------------------------------
            time_shift_amount = (
                np.random.randint(-time_shift, time_shift) if time_shift > 0 else 0
            )
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]

            # --- background noise ----------------------------------------------
            if use_background or sample["label"] == SILENCE_LABEL:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                if len(background_samples) <= desired_samples:
                    raise ValueError(
                        f"Background sample is too short! Need more than "
                        f"{desired_samples} samples but only "
                        f"{len(background_samples)} were found."
                    )
                background_offset = np.random.randint(
                    0, len(background_samples) - desired_samples
                )
                background_clipped = background_samples[
                    background_offset: background_offset + desired_samples
                ]
                background_reshaped = background_clipped.reshape([desired_samples, 1])

                if sample["label"] == SILENCE_LABEL:
                    background_volume = np.random.uniform(0, 1)
                elif np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0.0
            else:
                background_reshaped = np.zeros([desired_samples, 1], dtype=np.float32)
                background_volume = 0.0

            foreground_volume = 0.0 if sample["label"] == SILENCE_LABEL else 1.0

            # --- run processing (TF2 eager / tf.function) ----------------------
            output_tensor = self._process_sample(
                wav_filename=sample["file"],
                time_shift_padding=tf.constant(time_shift_padding, dtype=tf.int32),
                time_shift_offset=tf.constant(time_shift_offset, dtype=tf.int32),
                background_data=tf.constant(background_reshaped, dtype=tf.float32),
                background_volume=tf.constant(background_volume, dtype=tf.float32),
                foreground_volume=tf.constant(foreground_volume, dtype=tf.float32),
            )

            # --- optional TF2 summary ------------------------------------------
            if self.summary_writer_:
                with self.summary_writer_.as_default():
                    tf.summary.image(
                        "spectrogram_feature",
                        tf.expand_dims(tf.expand_dims(output_tensor, 0), -1),
                        step=i,
                    )

            data[step, :] = output_tensor.numpy().flatten()
            labels[step] = self.word_to_index[sample["label"]]
            step += 1

        return data, labels

    def get_features_for_wav(self, wav_filename, model_settings):
        """Applies feature extraction to a single WAV file (TF2 eager).

        TF2 version — ``sess`` parameter has been removed.

        Args:
            wav_filename:  Path to the input audio file.
            model_settings: Model configuration dict.

        Returns:
            Numpy array containing the generated features.
        """
        desired_samples = model_settings["desired_samples"]
        output_tensor = self._process_sample(
            wav_filename=wav_filename,
            time_shift_padding=tf.constant([[0, 0], [0, 0]], dtype=tf.int32),
            time_shift_offset=tf.constant([0, 0], dtype=tf.int32),
            background_data=tf.zeros([desired_samples, 1], dtype=tf.float32),
            background_volume=tf.constant(0.0, dtype=tf.float32),
            foreground_volume=tf.constant(1.0, dtype=tf.float32),
        )
        return output_tensor.numpy()

    def get_unprocessed_data(self, how_many, model_settings, mode):
        """Retrieve raw (unaugmented) sample data for the given partition.

        TF2 version — no session required.

        Args:
            how_many:       Desired number of samples (-1 = all).
            model_settings: Model configuration dict.
            mode:           'training', 'validation', or 'testing'.

        Returns:
            (data, labels) — numpy array of raw waveforms and string labels.
        """
        candidates = self.data_index[mode]
        sample_count = len(candidates) if how_many == -1 else how_many
        desired_samples = model_settings["desired_samples"]

        data = np.zeros((sample_count, desired_samples))
        labels = []

        for i in range(sample_count):
            sample_index = (
                i if how_many == -1 else np.random.randint(len(candidates))
            )
            sample = candidates[sample_index]

            # TF2: eager decode
            raw = tf.io.read_file(sample["file"])
            audio, _ = tf.audio.decode_wav(
                raw, desired_channels=1, desired_samples=desired_samples
            )
            volume = 0.0 if sample["label"] == SILENCE_LABEL else 1.0
            scaled = tf.multiply(audio, volume)

            data[i, :] = scaled.numpy().flatten()
            label_index = self.word_to_index[sample["label"]]
            labels.append(self.words_list[label_index])

        return data, labels