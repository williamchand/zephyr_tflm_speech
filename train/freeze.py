# freeze.py
# Copyright 2017 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0

import argparse
import logging
import os

import tensorflow as tf
import models
import input_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechCommandsInferenceModel(tf.Module):
    """Wraps the Keras model for TF SavedModel or TFLite export."""

    def __init__(self, keras_model, model_settings, preprocess):
        super().__init__(name='speech_commands_inference')
        self._model = keras_model

        # Flatten model_settings
        self.desired_samples = int(model_settings['desired_samples'])
        self.window_size = int(model_settings['window_size_samples'])
        self.window_stride = int(model_settings['window_stride_samples'])
        self.fingerprint_width = int(model_settings['fingerprint_width'])
        self.fingerprint_size = int(model_settings['fingerprint_size'])
        self.average_window_width = model_settings.get('average_window_width', -1)
        self.sample_rate = int(model_settings['sample_rate'])
        self.preprocess_mode = str(preprocess)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def infer(self, wav_data):
        # Decode WAV
        decoded = tf.audio.decode_wav(
            wav_data,
            desired_channels=1,
            desired_samples=self.desired_samples
        )
        audio_tensor = tf.squeeze(decoded.audio, axis=-1)

        # STFT
        spectrogram = tf.signal.stft(
            audio_tensor,
            frame_length=self.window_size,
            frame_step=self.window_stride,
            fft_length=self.window_size
        )
        spectrogram = tf.abs(spectrogram)

        # Preprocess features
        if self.preprocess_mode == 'average':
            fingerprint_input = tf.nn.pool(
                tf.expand_dims(spectrogram, -1),
                window_shape=[1, self.average_window_width],
                strides=[1, self.average_window_width],
                pooling_type='AVG',
                padding='SAME'
            )
        elif self.preprocess_mode in ('mfcc', 'micro'):
            fingerprint_input = tf.signal.mfccs_from_log_mel_spectrograms(
                tf.math.log(tf.maximum(spectrogram, 1e-6))
            )
        else:
            raise ValueError(f"Unknown preprocess mode {self.preprocess_mode}")

        reshaped = tf.reshape(fingerprint_input, [-1, self.fingerprint_size])
        logits = self._model(reshaped, training=False)
        return tf.nn.softmax(logits, name='labels_softmax')


# -----------------------------
# Model creation and export
# -----------------------------
def create_inference_model(FLAGS):
    words_list = input_data.prepare_words_list(FLAGS.wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        label_count=len(words_list),
        sample_rate=FLAGS.sample_rate,
        clip_duration_ms=FLAGS.clip_duration_ms,
        window_size_ms=FLAGS.window_size_ms,
        window_stride_ms=FLAGS.window_stride_ms,
        feature_bin_count=FLAGS.feature_bin_count,
        preprocess=FLAGS.preprocess
    )

    # Build Keras model
    dummy_input = tf.keras.Input(shape=(model_settings['fingerprint_size'],), name='fingerprint_input')
    result = models.create_model(
        dummy_input,
        model_settings,
        FLAGS.model_architecture,
        is_training=False
    )
    logits = result[0] if isinstance(result, tuple) else result
    keras_model = tf.keras.Model(inputs=dummy_input, outputs=logits)

    inference_module = SpeechCommandsInferenceModel(keras_model, model_settings, FLAGS.preprocess)
    return inference_module, model_settings


def save_saved_model(output_dir, inference_module):
    os.makedirs(output_dir, exist_ok=True)
    tf.saved_model.save(
        inference_module,
        output_dir,
        signatures={'serving_default': inference_module.infer}
    )
    logger.info("Saved SavedModel to %s", output_dir)


def save_tflite(output_file, inference_module, quantize=False):
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [inference_module.infer.get_concrete_function()],
        inference_module
    )
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info("Post-training quantization enabled.")
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    logger.info("Saved TFLite model to %s", output_file)


def main():
    inference_module, _ = create_inference_model(FLAGS)

    # Restore checkpoint if provided
    if FLAGS.start_checkpoint:
        checkpoint = tf.train.Checkpoint(model=inference_module._model)
        checkpoint.restore(FLAGS.start_checkpoint).expect_partial()
        logger.info("Restored checkpoint from %s", FLAGS.start_checkpoint)

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
        raise ValueError(f"Unknown save_format {save_format}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--clip_duration_ms', type=int, default=1000)
    parser.add_argument('--window_size_ms', type=float, default=30.0)
    parser.add_argument('--window_stride_ms', type=float, default=10.0)
    parser.add_argument('--feature_bin_count', type=int, default=40)
    parser.add_argument('--start_checkpoint', type=str, default='')
    parser.add_argument('--model_architecture', type=str, default='conv')
    parser.add_argument('--wanted_words', type=str,
                        default='yes,no,up,down,left,right,on,off,stop,go')
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--quantize', action='store_true', default=False)
    parser.add_argument('--preprocess', type=str, default='mfcc',
                        choices=['mfcc', 'micro', 'average'])
    parser.add_argument('--save_format', type=str, default='saved_model')
    FLAGS = parser.parse_args()

    main()