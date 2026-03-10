# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Converts a trained keyword-spotting model to TFLite and C source artefacts.

Supports two input paths
────────────────────────
  TF2 path  (train_tf2.py output)
    --keras_model_path=models/saved_model_tf2/model.keras
    For low_latency_svdf, pass the streaming SavedModel instead:
    --keras_model_path=models/saved_model_tf2/streaming

  TF1 path  (freeze.py output)
    --saved_model_path=models/saved_model_tf1

Produces three artefacts
────────────────────────
  <models_dir>/float_model.tflite   – float32, for accuracy benchmarking
  <models_dir>/model.tflite         – fully INT8-quantised (representative
                                      dataset from AudioProcessor)
  <models_dir>/model.cc             – C hex array for TF Lite Micro (TFLM)

Representative dataset
──────────────────────
Quantisation calibration uses the same AudioProcessor.get_data() pipeline
as training (mfcc / average / micro fingerprints) so the INT8 scale factors
are computed from the real distribution of activations, not synthetic noise.
This is the single most important factor for preserving post-quantisation
accuracy.

INT8 quantisation details
─────────────────────────
  • Input  : quantised to INT8  (full-integer mode)
  • Weights: quantised to INT8
  • Bias   : quantised to INT32
  • Output : quantised to INT8  (full-integer mode)
  quant_input_min/max (default 0.0 / 26.0 for micro preprocessing) define
  the input tensor scale.  Override with --quant_input_min / --quant_input_max
  for mfcc (−247 / 30) or average (0 / 127.5).

SVDF streaming note
───────────────────
The non-trainable rolling memory buffer in SVDFStreamingLayer uses
tf.Variable.assign(), which TFLite does not support.  The converter will
warn about this; the resulting .tflite is correct for batch inference but
the per-frame streaming state must be managed externally (pass the full
window each call).  True single-frame streaming requires a custom TFLM op.

Example
───────
    # TF2 Keras model (all architectures except SVDF streaming)
    python convert_tflite.py \\
        --keras_model_path=models/saved_model_tf2/model.keras \\
        --data_dir=dataset/ \\
        --wanted_words=yes,no,up,down,left,right,stop,go \\
        --preprocess=micro --window_stride=20 \\
        --models_dir=models

    # TF1 SavedModel (from freeze.py --save_format=saved_model)
    python convert_tflite.py \\
        --saved_model_path=models/saved_model_tf1 \\
        --data_dir=dataset/ \\
        --wanted_words=yes,no,up,down,left,right,stop,go \\
        --preprocess=micro --window_stride=20 \\
        --models_dir=models
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

import input_data
import models as models_tf1


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Convert trained keyword model to TFLite + C artefacts')

    # Input model — exactly one of these two must be provided
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        '--keras_model_path', type=str, default='',
        help='Path to a .keras file or SavedModel directory from train_tf2.py')
    src.add_argument(
        '--saved_model_path', type=str, default='',
        help='Path to a TF1 SavedModel directory from freeze.py '
             '(--save_format=saved_model)')

    # Output
    p.add_argument('--models_dir', type=str, default='models',
        help='Directory to write float_model.tflite, model.tflite, model.cc')

    # Dataset (for representative dataset calibration)
    p.add_argument('--data_url', type=str,
        default='https://storage.googleapis.com/download.tensorflow.org/'
                'data/speech_commands_v0.02.tar.gz')
    p.add_argument('--data_dir', type=str, default='dataset/')
    p.add_argument('--wanted_words', type=str,
        default='yes,no,up,down,left,right,stop,go')
    p.add_argument('--silence_percentage',   type=float, default=10.0)
    p.add_argument('--unknown_percentage',   type=float, default=10.0)
    p.add_argument('--validation_percentage', type=float, default=10.0)
    p.add_argument('--testing_percentage',   type=float, default=10.0)

    # Audio / preprocessing — must match the training run
    p.add_argument('--sample_rate',       type=int,   default=16000)
    p.add_argument('--clip_duration_ms',  type=int,   default=1000)
    p.add_argument('--window_size_ms',    type=float, default=30.0)
    p.add_argument('--window_stride',     type=float, default=20.0,
        help='Window stride in ms (same flag name as train_tf2.py)')
    p.add_argument('--feature_bin_count', type=int,   default=40)
    p.add_argument('--preprocess',        type=str,   default='micro',
        choices=['mfcc', 'average', 'micro'])

    # Quantisation range — must match the preprocessing mode
    # Defaults match preprocess=micro.  Override for mfcc (−247/30) or
    # average (0/127.5).
    p.add_argument('--quant_input_min', type=float, default=0.0,
        help='Minimum expected value of model input after preprocessing')
    p.add_argument('--quant_input_max', type=float, default=26.0,
        help='Maximum expected value of model input after preprocessing')

    # Calibration
    p.add_argument('--num_calibration_steps', type=int, default=100,
        help='Number of batches fed to the converter for INT8 calibration')

    return p.parse_args()


# ============================================================================
# Model loading
# ============================================================================

def load_model(args):
    """Loads the model from either a Keras file or a TF1 SavedModel directory.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A callable TF2 model (tf.keras.Model or a tf.saved_model loaded object).
    """
    if args.keras_model_path:
        print(f'Loading Keras model from: {args.keras_model_path}')
        # custom_objects needed for SVDFLayer / SVDFStreamingLayer if present
        try:
            from train_tf2 import SVDFLayer, SVDFStreamingLayer  # noqa
            custom_objects = {
                'SVDFLayer': SVDFLayer,
                'SVDFStreamingLayer': SVDFStreamingLayer,
            }
        except ImportError:
            custom_objects = {}

        model = tf.keras.models.load_model(
            args.keras_model_path,
            custom_objects=custom_objects or None,
            compile=False)
        print(f'  Loaded: {model.name}  '
              f'input={model.input_shape}  output={model.output_shape}')
        return model

    else:
        print(f'Loading TF1 SavedModel from: {args.saved_model_path}')
        model = tf.saved_model.load(args.saved_model_path)
        print('  Loaded TF1 SavedModel.')
        return model


# ============================================================================
# AudioProcessor + representative dataset
# ============================================================================

def build_audio_processor(args, model_settings):
    """Creates an AudioProcessor identical to the one used during training.

    Args:
        args:           Parsed CLI arguments.
        model_settings: Dict from models_tf1.prepare_model_settings().

    Returns:
        input_data.AudioProcessor instance.
    """
    return input_data.AudioProcessor(
        data_url              = args.data_url,
        data_dir              = args.data_dir,
        silence_percentage    = args.silence_percentage,
        unknown_percentage    = args.unknown_percentage,
        wanted_words          = args.wanted_words.split(','),
        validation_percentage = args.validation_percentage,
        testing_percentage    = args.testing_percentage,
        model_settings        = model_settings,
        summaries_dir         = '',
    )


def make_representative_dataset(audio_proc, model_settings,
                                 num_steps, quant_input_min,
                                 quant_input_max):
    """Returns a generator of calibration samples for INT8 quantisation.

    Uses AudioProcessor.get_data() (via a TF1 session) so that the
    calibration samples come from the exact same fingerprint pipeline as
    training.  Normalises to [quant_input_min, quant_input_max] before
    yielding so the converter sees the same dynamic range as inference.

    Each call yields one batch of shape [1, fingerprint_size] as required
    by TFLiteConverter.representative_dataset.

    Args:
        audio_proc:      Initialised AudioProcessor.
        model_settings:  Dict from models_tf1.prepare_model_settings().
        num_steps:       How many individual samples to yield.
        quant_input_min: Lower bound of the expected input range.
        quant_input_max: Upper bound of the expected input range.

    Yields:
        List containing one float32 tensor of shape [1, fingerprint_size].
    """
    batch_size       = 100
    fingerprint_size = model_settings['fingerprint_size']
    total_yielded    = 0

    with tf.compat.v1.Session(
            graph=tf.compat.v1.get_default_graph()) as sess:

        while total_yielded < num_steps:
            fingerprints, _ = audio_proc.get_data(
                how_many=batch_size,
                offset=0,
                model_settings=model_settings,
                background_frequency=0.0,
                background_volume_range=0.0,
                time_shift=0,
                mode='validation',
                sess=sess)

            # Normalise to the quantisation range so the converter's
            # scale/zero-point calculation is accurate.
            scale = quant_input_max - quant_input_min
            if scale > 0:
                fingerprints = (fingerprints - quant_input_min) / scale
            fingerprints = fingerprints.astype(np.float32)

            for i in range(len(fingerprints)):
                if total_yielded >= num_steps:
                    return
                sample = fingerprints[i].reshape(1, fingerprint_size)
                yield [sample]
                total_yielded += 1


# ============================================================================
# Float32 TFLite conversion
# ============================================================================

def convert_float(model, args, model_settings):
    """Converts the model to a float32 TFLite flatbuffer.

    No quantisation — useful for accuracy benchmarking and as a baseline
    before INT8 conversion.

    Args:
        model:          Loaded Keras model or SavedModel.
        args:           Parsed CLI arguments.
        model_settings: Dict from models_tf1.prepare_model_settings().

    Returns:
        Bytes of the float32 .tflite flatbuffer.
    """
    print('\n── Float32 conversion ──────────────────────────────────────────')

    if isinstance(model, tf.keras.Model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(
            args.saved_model_path)

    converter.optimizations           = []
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    print(f'  Size: {len(tflite_model) / 1024:.1f} KB')
    return tflite_model


# ============================================================================
# INT8 quantised TFLite conversion
# ============================================================================

def convert_int8(model, audio_proc, args, model_settings):
    """Converts the model to a fully INT8-quantised TFLite flatbuffer.

    Full-integer quantisation (inputs, weights, biases, outputs all INT8/INT32)
    using a representative dataset drawn from the real AudioProcessor pipeline.
    This is the format required by TF Lite Micro on microcontrollers.

    Args:
        model:          Loaded Keras model or SavedModel.
        audio_proc:     Initialised AudioProcessor.
        args:           Parsed CLI arguments.
        model_settings: Dict from models_tf1.prepare_model_settings().

    Returns:
        Bytes of the INT8-quantised .tflite flatbuffer.
    """
    print('\n── INT8 quantised conversion ───────────────────────────────────')
    print(f'  Calibrating with {args.num_calibration_steps} samples …')
    print(f'  Input range: [{args.quant_input_min}, {args.quant_input_max}]')

    if isinstance(model, tf.keras.Model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(
            args.saved_model_path)

    # Full-integer quantisation
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    # Representative dataset — real fingerprints from AudioProcessor
    def _representative_dataset():
        yield from make_representative_dataset(
            audio_proc, model_settings,
            args.num_calibration_steps,
            args.quant_input_min,
            args.quant_input_max)

    converter.representative_dataset = _representative_dataset

    tflite_model = converter.convert()
    print(f'  Size: {len(tflite_model) / 1024:.1f} KB')
    return tflite_model


# ============================================================================
# C hex array for TF Lite Micro
# ============================================================================

def convert_to_c_source(tflite_bytes, array_name='g_model'):
    """Converts a TFLite flatbuffer to a C source hex array.

    Produces a .cc file that can be compiled directly into a TFLM firmware
    image.  The array is declared as:
        alignas(8) const unsigned char g_model[] = { 0xXX, ... };
        const int g_model_len = NNNN;

    Args:
        tflite_bytes: bytes of the .tflite flatbuffer.
        array_name:   C identifier for the array (default: g_model).

    Returns:
        String containing the C source code.
    """
    hex_values = ', '.join(f'0x{b:02x}' for b in tflite_bytes)
    # Wrap at 12 bytes per line to keep lines under 80 chars
    bytes_list  = tflite_bytes
    lines       = []
    chunk_size  = 12
    for i in range(0, len(bytes_list), chunk_size):
        chunk = bytes_list[i:i + chunk_size]
        lines.append('    ' + ', '.join(f'0x{b:02x}' for b in chunk))

    return (
        '// Automatically generated by convert_tflite.py\n'
        '// DO NOT EDIT\n'
        '\n'
        '#include <cstdint>\n'
        '\n'
        f'alignas(8) const unsigned char {array_name}[] = {{\n'
        + ',\n'.join(lines) + '\n'
        '};\n'
        f'const int {array_name}_len = {len(tflite_bytes)};\n'
    )


# ============================================================================
# TFLite accuracy evaluation
# ============================================================================

def evaluate_tflite(tflite_bytes, audio_proc, model_settings, args,
                    label='float'):
    """Runs TFLite interpreter on the test set and prints accuracy.

    Loads the flatbuffer into tf.lite.Interpreter, feeds each test sample
    through it, and computes top-1 accuracy.  This validates that the
    conversion did not degrade accuracy beyond an acceptable threshold.

    Args:
        tflite_bytes:   bytes of the .tflite flatbuffer.
        audio_proc:     Initialised AudioProcessor.
        model_settings: Dict from models_tf1.prepare_model_settings().
        args:           Parsed CLI arguments.
        label:          String label for logging ('float' or 'int8').
    """
    print(f'\n── Evaluating {label} TFLite model on test set ──────────────')

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    fingerprint_size = model_settings['fingerprint_size']
    input_scale      = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    is_quantised     = input_details[0]['dtype'] == np.int8

    correct = 0
    total   = 0

    with tf.compat.v1.Session(
            graph=tf.compat.v1.get_default_graph()) as sess:

        set_size  = audio_proc.set_size('testing')
        batch_sz  = 100

        for offset in range(0, set_size, batch_sz):
            fingerprints, labels = audio_proc.get_data(
                how_many=batch_sz,
                offset=offset,
                model_settings=model_settings,
                background_frequency=0.0,
                background_volume_range=0.0,
                time_shift=0,
                mode='testing',
                sess=sess)

            for i in range(len(fingerprints)):
                sample = fingerprints[i].reshape(1, fingerprint_size)

                if is_quantised:
                    # Scale float fingerprint to INT8 using the tensor's
                    # quantisation parameters from the converter.
                    if input_scale > 0:
                        sample = sample / input_scale + input_zero_point
                    sample = np.clip(sample, -128, 127).astype(np.int8)

                interpreter.set_tensor(input_details[0]['index'], sample)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                if np.argmax(output) == int(labels[i]):
                    correct += 1
                total += 1

    accuracy = 100.0 * correct / max(total, 1)
    print(f'  {label} accuracy: {accuracy:.2f}%  ({correct}/{total})')
    return accuracy


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    # ── 1. Model settings (must match training) ───────────────────────────
    wanted_words_list = args.wanted_words.split(',')
    words_list        = input_data.prepare_words_list(wanted_words_list)

    model_settings = models_tf1.prepare_model_settings(
        label_count       = len(words_list),
        sample_rate       = args.sample_rate,
        clip_duration_ms  = args.clip_duration_ms,
        window_size_ms    = args.window_size_ms,
        window_stride_ms  = args.window_stride,
        feature_bin_count = args.feature_bin_count,
        preprocess        = args.preprocess,
    )
    print(f'fingerprint_size={model_settings["fingerprint_size"]},  '
          f'preprocess={args.preprocess}')

    # ── 2. Load model ──────────────────────────────────────────────────────
    model = load_model(args)

    # ── 3. AudioProcessor (same config as training) ───────────────────────
    print('\nInitialising AudioProcessor for calibration / evaluation …')
    audio_proc = build_audio_processor(args, model_settings)

    # ── 4. Float32 TFLite ─────────────────────────────────────────────────
    float_tflite = convert_float(model, args, model_settings)

    float_path = os.path.join(args.models_dir, 'float_model.tflite')
    with open(float_path, 'wb') as f:
        f.write(float_tflite)
    print(f'  Written: {float_path}')

    float_acc = evaluate_tflite(
        float_tflite, audio_proc, model_settings, args, label='float32')

    # ── 5. INT8 quantised TFLite ──────────────────────────────────────────
    int8_tflite = convert_int8(model, audio_proc, args, model_settings)

    int8_path = os.path.join(args.models_dir, 'model.tflite')
    with open(int8_path, 'wb') as f:
        f.write(int8_tflite)
    print(f'  Written: {int8_path}')

    int8_acc = evaluate_tflite(
        int8_tflite, audio_proc, model_settings, args, label='int8')

    # ── 6. C hex array (.cc) ──────────────────────────────────────────────
    print('\n── C source array (TFLM) ────────────────────────────────────────')
    c_src = convert_to_c_source(int8_tflite, array_name='g_model')

    cc_path = os.path.join(args.models_dir, 'model.cc')
    with open(cc_path, 'w') as f:
        f.write(c_src)
    print(f'  Written: {cc_path}  ({len(int8_tflite)} bytes)')

    # ── 7. Summary ────────────────────────────────────────────────────────
    print('\n── Conversion summary ───────────────────────────────────────────')
    print(f'  float32 TFLite : {float_path}')
    print(f'    size         : {len(float_tflite) / 1024:.1f} KB')
    print(f'    accuracy     : {float_acc:.2f}%')
    print(f'  INT8 TFLite    : {int8_path}')
    print(f'    size         : {len(int8_tflite) / 1024:.1f} KB')
    print(f'    accuracy     : {int8_acc:.2f}%')
    print(f'    accuracy drop: {float_acc - int8_acc:.2f}%')
    print(f'  C source array : {cc_path}')
    print(f'  Compression    : {len(float_tflite) / max(len(int8_tflite), 1):.1f}× '
          f'({len(float_tflite) / 1024:.1f} KB → '
          f'{len(int8_tflite) / 1024:.1f} KB)')

    if float_acc - int8_acc > 2.0:
        print('\n  ⚠  Accuracy drop > 2%.  Consider:')
        print('     • Increasing --num_calibration_steps (try 500–1000)')
        print('     • Checking --quant_input_min/max match your preprocessing mode')
        print('     • preprocess=micro expects min=0.0,  max=26.0')
        print('     • preprocess=mfcc  expects min=-247.0, max=30.0')
        print('     • preprocess=average expects min=0.0, max=127.5')

    print('\nDone.')


if __name__ == '__main__':
    main()