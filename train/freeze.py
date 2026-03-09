# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================

r"""Converts a trained checkpoint into a frozen model for mobile inference (TF2).

Example usage:
python freeze.py \
    --sample_rate=16000 --feature_bin_count=40 \
    --model_architecture=conv \
    --start_checkpoint=/tmp/checkpoint.ckpt \
    --output_file=/tmp/my_model \
    --save_format=saved_model
"""

import argparse
import logging
import os
import tensorflow as tf
import input_data
import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from tensorflow.lite.experimental.microfrontend.python.ops import (
        audio_microfrontend_op as frontend_op
    )
except ImportError:
    frontend_op = None

FLAGS = None

# ---------------------------------------------------------------------------
# Inference Module
# ---------------------------------------------------------------------------

class SpeechCommandsInferenceModel(tf.Module):
    """Wraps a Keras model with WAV bytes -> softmax tf.function."""

    def __init__(self, keras_model, model_settings, preprocess):
        super().__init__(name='speech_commands_inference')
        self._model = keras_model

        # Extract primitives from dict to avoid Python 3.12 _DictWrapper errors
        desired_samples = int(model_settings['desired_samples'])
        window_size = int(model_settings['window_size_samples'])
        window_stride = int(model_settings['window_stride_samples'])
        fingerprint_width = int(model_settings['fingerprint_width'])
        fingerprint_size = int(model_settings['fingerprint_size'])
        average_window_width = int(model_settings.get('average_window_width', -1))
        sample_rate = int(model_settings['sample_rate'])
        preprocess_mode = str(preprocess)

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def infer(wav_data):
            # Decode WAV bytes
            decoded = tf.audio.decode_wav(
                wav_data,
                desired_channels=1,
                desired_samples=desired_samples
            )

            audio_tensor = tf.squeeze(decoded.audio, axis=-1)

            # Compute spectrogram
            spectrogram = tf.signal.stft(
                audio_tensor,
                frame_length=window_size,
                frame_step=window_stride,
                fft_length=window_size
            )
            spectrogram = tf.abs(spectrogram)

            # Preprocess
            if preprocess_mode == 'average':
                fingerprint_input = tf.nn.pool(
                    input=tf.expand_dims(spectrogram, -1),
                    window_shape=[1, average_window_width],
                    strides=[1, average_window_width],
                    pooling_type='AVG',
                    padding='SAME'
                )
            elif preprocess_mode == 'mfcc':
                fingerprint_input = tf.signal.mfccs_from_log_mel_spectrograms(
                    tf.math.log(tf.maximum(spectrogram, 1e-6))
                )
                fingerprint_input = tf.reshape(fingerprint_input, [-1, fingerprint_size])
            elif preprocess_mode == 'micro':
                if not frontend_op:
                    raise RuntimeError('Micro frontend op not available.')
                int16_input = tf.cast(audio_tensor * 32767, tf.int16)
                fingerprint_input = frontend_op.audio_microfrontend(
                    int16_input,
                    sample_rate=sample_rate,
                    window_size=(window_size * 1000) / sample_rate,
                    window_step=(window_stride * 1000) / sample_rate,
                    num_channels=fingerprint_width,
                    out_scale=1,
                    out_type=tf.float32
                )
                fingerprint_input = tf.reshape(fingerprint_input, [-1, fingerprint_size])
            else:
                raise ValueError(f'Unknown preprocess mode "{preprocess_mode}"')

            # Forward pass through Keras model
            logits = self._model(fingerprint_input, training=False)
            softmax = tf.nn.softmax(logits, name='labels_softmax')
            return softmax

        self.infer = infer


# ---------------------------------------------------------------------------
# Build inference graph + load weights
# ---------------------------------------------------------------------------

def create_inference_graph(
        wanted_words, sample_rate, clip_duration_ms, clip_stride_ms,
        window_size_ms, window_stride_ms, feature_bin_count,
        model_architecture, preprocess):
    words_list = input_data.prepare_words_list(wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms,
        window_size_ms, window_stride_ms, feature_bin_count, preprocess)

    # Keras functional API
    fingerprint_size = model_settings['fingerprint_size']
    inputs = tf.keras.Input(shape=(fingerprint_size,), name='fingerprint_input')
    result = models.create_model(
        inputs, model_settings, model_architecture, is_training=False)
    logits = result[0] if isinstance(result, tuple) else result
    keras_model = tf.keras.Model(inputs=inputs, outputs=logits)

    inference_module = SpeechCommandsInferenceModel(keras_model, model_settings, preprocess)
    return inference_module, model_settings


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_saved_model(output_path, inference_module):
    os.makedirs(output_path, exist_ok=True)
    tf.saved_model.save(
        inference_module,
        output_path,
        signatures={'serving_default': inference_module.infer}
    )
    logger.info('Saved SavedModel to %s', output_path)


def save_tflite(output_file, inference_module, quantize=False):
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [inference_module.infer.get_concrete_function()]
    )
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info('Enabled post-training dynamic-range quantization.')
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    logger.info('Saved TFLite model to %s', output_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    inference_module, _ = create_inference_graph(
        FLAGS.wanted_words,
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.clip_stride_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.feature_bin_count,
        FLAGS.model_architecture,
        FLAGS.preprocess
    )

    # Restore checkpoint if provided
    if FLAGS.start_checkpoint:
        checkpoint = tf.train.Checkpoint(model=inference_module._model)
        status = checkpoint.restore(FLAGS.start_checkpoint)
        status.expect_partial()
        logger.info('Restored checkpoint from %s', FLAGS.start_checkpoint)

    # Export model
    save_format = FLAGS.save_format.lower()
    if save_format == 'saved_model':
        save_saved_model(FLAGS.output_file, inference_module)
    elif save_format in ('graph_def', 'tflite'):
        output_file = FLAGS.output_file
        if not output_file.endswith('.tflite'):
            output_file += '.tflite'
        save_tflite(output_file, inference_module, quantize=FLAGS.quantize)
    else:
        raise ValueError(f'Unknown save_format "{FLAGS.save_format}"')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--clip_duration_ms', type=int, default=1000)
    parser.add_argument('--clip_stride_ms', type=int, default=30)
    parser.add_argument('--window_size_ms', type=float, default=30.0)
    parser.add_argument('--window_stride_ms', type=float, default=10.0)
    parser.add_argument('--feature_bin_count', type=int, default=40)
    parser.add_argument('--start_checkpoint', type=str, default='')
    parser.add_argument('--model_architecture', type=str, default='conv')
    parser.add_argument('--wanted_words', type=str,
                        default='yes,no,up,down,left,right,on,off,stop,go')
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--quantize', action='store_true', default=False)
    parser.add_argument('--preprocess', type=str, default='mfcc')
    parser.add_argument('--save_format', type=str, default='saved_model')
    FLAGS = parser.parse_args()
    main()