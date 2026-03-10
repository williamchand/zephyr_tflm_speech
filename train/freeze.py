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

Once you've trained a model using the `train.py` script, you can use this tool
to convert it into a binary GraphDef file that can be loaded into the Android,
iOS, or Raspberry Pi example code. Here's an example of how to run it:

bazel run tensorflow/examples/speech_commands/freeze -- \
--sample_rate=16000 --dct_coefficient_count=40 --window_size_ms=20 \
--window_stride_ms=10 --clip_duration_ms=1000 \
--model_architecture=conv \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-1300 \
--output_file=/tmp/my_frozen_graph.pb

One thing to watch out for is that you need to pass in the same arguments for
`sample_rate` and other command line variables here as you did for the training
script.

The resulting graph has an input for WAV-encoded data named 'wav_data', one for
raw PCM data (as floats in the range -1.0 to 1.0) called 'decoded_sample_data',
and the output is called 'labels_softmax'.

Refactoring audit (original → this version)
────────────────────────────────────────────
CHANGED (cosmetic — safer, no functional impact):
  tf.compat.v1.logging.info  →  tf.get_logger().info()
  tf.compat.v1.app.run       →  argparse + direct main() call

NOT CHANGED (functional — must remain TF1):
  tf.compat.v1.placeholder
      Required: wav_data is a graph-mode string placeholder at the
      entry point of the freeze inference graph.

  tf.compat.v1.InteractiveSession
      Required: the freeze workflow is inherently graph-mode — build
      graph → load checkpoint variables → run
      convert_variables_to_constants → save.  There is no TF2
      equivalent for frozen .pb graphs.

  tf.compat.v1.graph_util.convert_variables_to_constants
      Required: TF2 has no public replacement for this.  The frozen
      .pb format is the target output of this script.

  tf.compat.v1.saved_model.* (builder, utils, signature_def_utils …)
      Required: saving a TF1 session graph as a SavedModel cannot use
      the TF2 tf.saved_model.save() API because there is no
      tf.function wrapping the session.  The TF1 builder API is the
      only path that preserves the graph + checkpoint variables.

  tf.contrib.quantize.*
      Dead code — already guards itself with a clear error message
      telling the user to install TF<=1.15.  Not touched.
"""
import argparse
import os.path
import sys

import tensorflow as tf

import input_data
import models
from tensorflow.python.ops import gen_audio_ops as audio_ops

# Microfrontend op is only available in Bazel builds.
try:
    from tensorflow.lite.experimental.microfrontend.python.ops \
        import audio_microfrontend_op as frontend_op
except ImportError:
    frontend_op = None

FLAGS = None


def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           clip_stride_ms, window_size_ms, window_stride_ms,
                           feature_bin_count, model_architecture, preprocess):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.

    Args:
        wanted_words:        Comma-separated list of words to recognise.
        sample_rate:         Audio samples per second.
        clip_duration_ms:    Length of each audio clip in ms.
        clip_stride_ms:      How often to run recognition (streaming).
        window_size_ms:      Spectrogram window duration in ms.
        window_stride_ms:    Spectrogram window stride in ms.
        feature_bin_count:   Number of frequency bands.
        model_architecture:  Architecture name string.
        preprocess:          'mfcc', 'average', or 'micro'.

    Returns:
        (input_tensor, output_tensor) pair.

    Raises:
        Exception: If preprocessing mode is not recognised or micro
                   frontend is unavailable.
    """
    words_list     = input_data.prepare_words_list(wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms,
        window_size_ms, window_stride_ms, feature_bin_count, preprocess)
    runtime_settings = {'clip_stride_ms': clip_stride_ms}

    wav_data_placeholder = tf.compat.v1.placeholder(
        tf.string, [], name='wav_data')
    decoded_sample_data  = tf.audio.decode_wav(
        wav_data_placeholder,
        desired_channels=1,
        desired_samples=model_settings['desired_samples'],
        name='decoded_sample_data')
    spectrogram = audio_ops.audio_spectrogram(
        decoded_sample_data.audio,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)

    if preprocess == 'average':
        fingerprint_input = tf.nn.pool(
            input=tf.expand_dims(spectrogram, -1),
            window_shape=[1, model_settings['average_window_width']],
            strides=[1, model_settings['average_window_width']],
            pooling_type='AVG',
            padding='SAME')
    elif preprocess == 'mfcc':
        fingerprint_input = audio_ops.mfcc(
            spectrogram,
            sample_rate,
            dct_coefficient_count=model_settings['fingerprint_width'])
    elif preprocess == 'micro':
        if not frontend_op:
            raise Exception(
                'Micro frontend op is currently not available when running '
                'TensorFlow directly from Python. Build with Bazel:\n'
                '  bazel run tensorflow/examples/speech_commands:freeze_graph')
        sample_rate    = model_settings['sample_rate']
        window_size_ms = (model_settings['window_size_samples']
                          * 1000) / sample_rate
        window_step_ms = (model_settings['window_stride_samples']
                          * 1000) / sample_rate
        int16_input    = tf.cast(
            tf.multiply(decoded_sample_data.audio, 32767), tf.int16)
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=sample_rate,
            window_size=window_size_ms,
            window_step=window_step_ms,
            num_channels=model_settings['fingerprint_width'],
            out_scale=1,
            out_type=tf.float32)
        fingerprint_input = tf.multiply(micro_frontend, (10.0 / 256.0))
    else:
        raise Exception(
            'Unknown preprocess mode "%s" (should be "mfcc", "average", or '
            '"micro")' % preprocess)

    fingerprint_size = model_settings['fingerprint_size']
    reshaped_input   = tf.reshape(fingerprint_input, [-1, fingerprint_size])

    logits = models.create_model(
        reshaped_input, model_settings, model_architecture,
        is_training=False, runtime_settings=runtime_settings)

    softmax = tf.nn.softmax(logits, name='labels_softmax')
    return reshaped_input, softmax


def save_graph_def(file_name, frozen_graph_def):
    """Writes a GraphDef protobuf to disk as a binary .pb file.

    Args:
        file_name:         Output path.
        frozen_graph_def:  GraphDef proto to save.
    """
    tf.io.write_graph(
        frozen_graph_def,
        os.path.dirname(file_name),
        os.path.basename(file_name),
        as_text=False)
    # Refactored: tf.compat.v1.logging.info → tf.get_logger().info
    tf.get_logger().info('Saved frozen graph to %s', file_name)


def save_saved_model(file_name, sess, input_tensor, output_tensor):
    """Writes a TF1 SavedModel to disk.

    NOT refactored to tf.saved_model.save(): the TF2 API requires a
    tf.function-wrapped callable, not a Session.  The TF1 SavedModelBuilder
    is the only correct path when saving from a TF1 graph session.

    Args:
        file_name:      Output directory path.
        sess:           Active tf.compat.v1.Session.
        input_tensor:   Tensor describing the model input.
        output_tensor:  Tensor describing the model output.
    """
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(file_name)
    tensor_info_inputs  = {
        'input': tf.compat.v1.saved_model.utils.build_tensor_info(input_tensor)
    }
    tensor_info_outputs = {
        'output': tf.compat.v1.saved_model.utils.build_tensor_info(output_tensor)
    }
    signature = (
        tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs=tensor_info_inputs,
            outputs=tensor_info_outputs,
            method_name=(tf.compat.v1.saved_model.signature_constants
                         .PREDICT_METHOD_NAME)))
    builder.add_meta_graph_and_variables(
        sess,
        [tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.compat.v1.saved_model.signature_constants
            .DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature,
        },
    )
    builder.save()


def main(_):
    if FLAGS.quantize:
        try:
            _ = tf.contrib
        except AttributeError as e:
            msg = e.args[0]
            msg += (
                '\n\n The --quantize option still requires contrib, which is '
                'not part of TensorFlow 2.0. Please install a previous '
                'version:\n    `pip install tensorflow<=1.15`')
            e.args = (msg,)
            raise e

    sess = tf.compat.v1.InteractiveSession()
    input_tensor, output_tensor = create_inference_graph(
        FLAGS.wanted_words, FLAGS.sample_rate, FLAGS.clip_duration_ms,
        FLAGS.clip_stride_ms, FLAGS.window_size_ms, FLAGS.window_stride_ms,
        FLAGS.feature_bin_count, FLAGS.model_architecture, FLAGS.preprocess)

    if FLAGS.quantize:
        tf.contrib.quantize.create_eval_graph()

    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['labels_softmax'])

    if FLAGS.save_format == 'graph_def':
        save_graph_def(FLAGS.output_file, frozen_graph_def)
    elif FLAGS.save_format == 'saved_model':
        save_saved_model(FLAGS.output_file, sess, input_tensor, output_tensor)
    else:
        raise Exception(
            'Unknown save format "%s" (should be "graph_def" or '
            '"saved_model")' % FLAGS.save_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate',      type=int,   default=16000)
    parser.add_argument('--clip_duration_ms', type=int,   default=1000)
    parser.add_argument('--clip_stride_ms',   type=int,   default=30)
    parser.add_argument('--window_size_ms',   type=float, default=30.0)
    parser.add_argument('--window_stride_ms', type=float, default=10.0)
    parser.add_argument('--feature_bin_count',type=int,   default=40)
    parser.add_argument('--start_checkpoint', type=str,   default='')
    parser.add_argument('--model_architecture', type=str, default='conv')
    parser.add_argument('--wanted_words',     type=str,
        default='yes,no,up,down,left,right,on,off,stop,go')
    parser.add_argument('--output_file',      type=str,
        help='Where to save the frozen graph.')
    parser.add_argument('--quantize',         type=bool,  default=False)
    parser.add_argument('--preprocess',       type=str,   default='mfcc',
        help='Spectrogram processing mode: "mfcc", "average", or "micro"')
    parser.add_argument('--save_format',      type=str,   default='graph_def',
        help='Output format: "graph_def" or "saved_model"')
    FLAGS, unparsed = parser.parse_known_args()

    # Refactored: tf.compat.v1.app.run → direct call; argparse handles argv.
    main(None)