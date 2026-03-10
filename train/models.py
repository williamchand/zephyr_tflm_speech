# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Audio model settings helper — no TensorFlow dependency.

Extracted from models.py as part of the TF1 deprecation.  All TF2 scripts
(train_tf2.py, convert_tflite.py, evaluate.py, input_data.py) now import
from here instead of from models.py.

models.py is retained unchanged for the legacy TF1 pipeline (train.py,
freeze.py) but should not be imported by any TF2 code.
"""
import math


def _next_power_of_two(x):
    """Returns the smallest power of two that is >= x."""
    return 1 if x == 0 else 2 ** (int(x) - 1).bit_length()


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                            window_size_ms, window_stride_ms,
                            feature_bin_count, preprocess):
    """Calculates common settings needed for all models.

    Pure Python — no TensorFlow import.  Identical logic to
    models.prepare_model_settings(); kept here so TF2 scripts have no
    dependency on the TF1 models.py file.

    Args:
        label_count:       Number of output classes (including silence and
                           unknown).
        sample_rate:       Audio samples per second (typically 16000).
        clip_duration_ms:  Length of each audio clip in milliseconds.
        window_size_ms:    STFT window duration in milliseconds.
        window_stride_ms:  STFT window stride in milliseconds.
        feature_bin_count: Number of frequency bins / MFCC coefficients.
        preprocess:        Feature extraction mode: 'mfcc', 'average', or
                           'micro'.

    Returns:
        Dict with keys:
            desired_samples, window_size_samples, window_stride_samples,
            spectrogram_length, fingerprint_width, fingerprint_size,
            label_count, sample_rate, preprocess, average_window_width.

    Raises:
        ValueError: If preprocess is not 'mfcc', 'average', or 'micro'.
    """
    desired_samples       = int(sample_rate * clip_duration_ms  / 1000)
    window_size_samples   = int(sample_rate * window_size_ms    / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms  / 1000)

    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

    if preprocess == 'average':
        fft_bin_count        = 1 + (_next_power_of_two(window_size_samples) / 2)
        average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
        fingerprint_width    = int(math.ceil(fft_bin_count  / average_window_width))
    elif preprocess in ('mfcc', 'micro'):
        average_window_width = -1
        fingerprint_width    = feature_bin_count
    else:
        raise ValueError(
            'Unknown preprocess mode "%s" (should be "mfcc", "average", or '
            '"micro")' % preprocess)

    fingerprint_size = fingerprint_width * spectrogram_length

    return {
        'desired_samples':       desired_samples,
        'window_size_samples':   window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length':    spectrogram_length,
        'fingerprint_width':     fingerprint_width,
        'fingerprint_size':      fingerprint_size,
        'label_count':           label_count,
        'sample_rate':           sample_rate,
        'preprocess':            preprocess,
        'average_window_width':  average_window_width,
    }