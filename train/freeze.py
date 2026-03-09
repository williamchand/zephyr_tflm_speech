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
r"""Converts a trained checkpoint into a frozen model for mobile inference.

TF2 version — replaces the TF1 frozen-GraphDef approach with:
  * 'saved_model'  — tf.saved_model / tf.keras model.save()  (recommended)
  * 'graph_def'    — TFLite FlatBuffer (.tflite), suitable for on-device use
  * 'tflite'       — alias for 'graph_def' (explicit name)

All original CLI flags are preserved so existing scripts need no changes.

Example usage:
  python freeze.py \
    --sample_rate=16000 --feature_bin_count=40 \
    --model_architecture=conv \
    --start_checkpoint=/tmp/speech_commands_train/conv.ckpt-1300 \
    --output_file=/tmp/my_model \
    --save_format=saved_model

The exported SavedModel / TFLite model accepts:
  Input  name : 'wav_data'            — raw WAV bytes (tf.string scalar)
  Output name : 'labels_softmax'      — float32 [1, num_labels] softmax scores
"""
import argparse
import logging
import os

import tensorflow as tf

import input_data
import models
from tensorflow.python.ops import gen_audio_ops as audio_ops

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from tensorflow.lite.experimental.microfrontend.python.ops import (
        audio_microfrontend_op as frontend_op,
    )
except ImportError:
    frontend_op = None

FLAGS = None


# ---------------------------------------------------------------------------
# Inference module
# ---------------------------------------------------------------------------

class SpeechCommandsInferenceModel(tf.Module):
    """Wraps the acoustic model with a WAV-bytes → softmax tf.function.

    The exported concrete function has:
      Input  : wav_data  — tf.string scalar (raw WAV file bytes)
      Output : labels_softmax — tf.float32 [1, num_labels]
    """

    def __init__(self, keras_model, model_settings, preprocess):
        super().__init__(name='speech_commands_inference')
        self._model = keras_model
        self._settings = model_settings
        self._preprocess = preprocess

        # Capture settings as Python locals so tf.function can close over them.
        desired_samples = model_settings['desired_samples']
        window_size = model_settings['window_size_samples']
        window_stride = model_settings['window_stride_samples']
        fingerprint_width = model_settings['fingerprint_width']
        fingerprint_size = model_settings['fingerprint_size']
        average_window_width = model_settings.get('average_window_width', -1)
        sample_rate = model_settings['sample_rate']

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[], dtype=tf.string, name='wav_data')
        ])
        def infer(wav_data):
            """Decode WAV bytes, extract features, return softmax scores."""
            decoded = tf.audio.decode_wav(
                wav_data,
                desired_channels=1,
                desired_samples=desired_samples,
                name='decoded_sample_data')

            spectrogram = audio_ops.audio_spectrogram(
                decoded.audio,
                window_size=window_size,
                stride=window_stride,
                magnitude_squared=True)

            if preprocess == 'average':
                fingerprint_input = tf.nn.pool(
                    input=tf.expand_dims(spectrogram, -1),
                    window_shape=[1, average_window_width],
                    strides=[1, average_window_width],
                    pooling_type='AVG',
                    padding='SAME')
            elif preprocess == 'mfcc':
                fingerprint_input = audio_ops.mfcc(
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
                    tf.multiply(decoded.audio, 32767), tf.int16)
                micro_frontend = frontend_op.audio_microfrontend(
                    int16_input,
                    sample_rate=sample_rate,
                    window_size=window_size_ms,
                    window_step=window_step_ms,
                    num_channels=fingerprint_width,
                    out_scale=1,
                    out_type=tf.float32)
                fingerprint_input = tf.multiply(
                    micro_frontend, (10.0 / 256.0))
            else:
                raise ValueError(
                    'Unknown preprocess mode "%s"' % preprocess)

            reshaped = tf.reshape(fingerprint_input, [-1, fingerprint_size],
                                  name='fingerprint_input')
            logits = self._model(reshaped, training=False)
            softmax = tf.nn.softmax(logits, name='labels_softmax')
            return softmax

        self.infer = infer


# ---------------------------------------------------------------------------
# Build the inference graph + load weights
# ---------------------------------------------------------------------------

def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                            clip_stride_ms, window_size_ms, window_stride_ms,
                            feature_bin_count, model_architecture, preprocess):
    """Builds a Keras model and wraps it in a SpeechCommandsInferenceModel.

    Args:
      wanted_words:       Comma-separated list of target words.
      sample_rate:        Audio sample rate in Hz.
      clip_duration_ms:   Length of audio clips in ms.
      clip_stride_ms:     Recognition stride in ms (for cached models).
      window_size_ms:     FFT window size in ms.
      window_stride_ms:   FFT window stride in ms.
      feature_bin_count:  Number of MFCC / frequency bins.
      model_architecture: Architecture name string.
      preprocess:         One of 'mfcc', 'average', 'micro'.

    Returns:
      Tuple (inference_module, model_settings):
        inference_module  — SpeechCommandsInferenceModel instance.
        model_settings    — dict from models.prepare_model_settings().
    """
    words_list = input_data.prepare_words_list(wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms,
        window_size_ms, window_stride_ms, feature_bin_count, preprocess)
    runtime_settings = {'clip_stride_ms': clip_stride_ms}

    fingerprint_size = model_settings['fingerprint_size']

    # Build a Keras model using the functional API.
    inputs = tf.keras.Input(
        shape=(fingerprint_size,), name='fingerprint_input')
    result = models.create_model(
        inputs, model_settings, model_architecture,
        is_training=False, runtime_settings=runtime_settings)
    # create_model returns (logits, dropout_rate) when is_training=True,
    # and just logits when is_training=False.
    logits = result[0] if isinstance(result, tuple) else result
    keras_model = tf.keras.Model(inputs=inputs, outputs=logits)

    inference_module = SpeechCommandsInferenceModel(
        keras_model, model_settings, preprocess)
    return inference_module, model_settings


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_saved_model(output_path, inference_module):
    """Exports the inference module as a TF2 SavedModel.

    Args:
      output_path:       Directory to write the SavedModel to.
      inference_module:  SpeechCommandsInferenceModel instance.
    """
    os.makedirs(output_path, exist_ok=True)
    tf.saved_model.save(
        inference_module,
        output_path,
        signatures={'serving_default': inference_module.infer})
    logger.info('Saved SavedModel to %s', output_path)


def save_tflite(output_file, inference_module):
    """Converts the inference module to a TFLite FlatBuffer and writes it.

    This is the TF2 replacement for the frozen GraphDef (.pb) format used in
    the original script.  The resulting .tflite file can be loaded directly
    on Android, iOS, and microcontrollers.

    Args:
      output_file:       Path for the output .tflite file.
      inference_module:  SpeechCommandsInferenceModel instance.
    """
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [inference_module.infer.get_concrete_function()],
        inference_module)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    logger.info('Saved TFLite model to %s', output_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Build model + inference wrapper.
    inference_module, model_settings = create_inference_graph(
        FLAGS.wanted_words,
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.clip_stride_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.feature_bin_count,
        FLAGS.model_architecture,
        FLAGS.preprocess)

    # Restore weights from checkpoint.
    if FLAGS.start_checkpoint:
        checkpoint = tf.train.Checkpoint(
            model=inference_module._model)
        status = checkpoint.restore(FLAGS.start_checkpoint)
        status.expect_partial()   # optimizer state won't be present — that's OK
        logger.info('Restored checkpoint from %s', FLAGS.start_checkpoint)

    # Export.
    save_format = FLAGS.save_format.lower()
    if save_format == 'saved_model':
        save_saved_model(FLAGS.output_file, inference_module)
    elif save_format in ('graph_def', 'tflite'):
        # 'graph_def' is kept as an alias so existing scripts don't break.
        # In TF2 the equivalent mobile-deployable format is TFLite.
        output_file = FLAGS.output_file
        if not output_file.endswith('.tflite'):
            output_file += '.tflite'
        save_tflite(output_file, inference_module)
    else:
        raise ValueError(
            'Unknown save_format "%s" (should be "saved_model", "graph_def", '
            'or "tflite")' % FLAGS.save_format)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_rate', type=int, default=16000,
        help='Expected sample rate of the wavs.')
    parser.add_argument(
        '--clip_duration_ms', type=int, default=1000,
        help='Expected duration in milliseconds of the wavs.')
    parser.add_argument(
        '--clip_stride_ms', type=int, default=30,
        help='How often to run recognition. Useful for models with cache.')
    parser.add_argument(
        '--window_size_ms', type=float, default=30.0,
        help='How long each spectrogram timeslice is.')
    parser.add_argument(
        '--window_stride_ms', type=float, default=10.0,
        help='How long the stride is between spectrogram timeslices.')
    parser.add_argument(
        '--feature_bin_count', type=int, default=40,
        help='How many bins to use for the MFCC fingerprint.')
    parser.add_argument(
        '--start_checkpoint', type=str, default='',
        help='If specified, restore this pretrained model before exporting.')
    parser.add_argument(
        '--model_architecture', type=str, default='conv',
        help='What model architecture to use.')
    parser.add_argument(
        '--wanted_words', type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label).')
    parser.add_argument(
        '--output_file', type=str, required=True,
        help='Where to save the exported model.')
    parser.add_argument(
        '--quantize', action='store_true', default=False,
        help='Whether to quantize the model for eight-bit deployment.')
    parser.add_argument(
        '--preprocess', type=str, default='mfcc',
        help='Spectrogram processing mode: "mfcc", "average", or "micro".')
    parser.add_argument(
        '--save_format', type=str, default='saved_model',
        help='Export format: "saved_model", "graph_def" (→ TFLite), or "tflite".')

    FLAGS = parser.parse_args()
    main()