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
"""Keyword-spotting evaluation with ring-buffer streaming inference.

Architecture
────────────
This file is the Python counterpart of the reference TFLM file
(micro_speech_streaming.py).  The ring-buffer design is identical:

    ring_buffer  shape [49, 40]  =  _FEATURES_SHAPE = (49, 40)
    shift_ring_buffer()          =  reference shift_ring_buffer()
    predict()                    =  reference predict()  (all backends)
    generate_features()          =  reference generate_features()
                                    but uses input_data.preprocess_audio_tf2()
                                    instead of audio_preprocessor TFLM op

Connection to train_tf2.py
───────────────────────────
  • model_settings_lib.prepare_model_settings()  ── same function as training
  • input_data.AudioProcessor.get_data()         ── same preprocessing path
  • input_data.preprocess_audio_tf2()            ── produces [spec_len, feat_w]
                                                    i.e. one ring buffer's worth
  • SVDFLayer / SVDFStreamingLayer custom objects imported from train_tf2.py

Two modes
─────────
  Batch mode  (default, no --wav_file)
      AudioProcessor.get_data() returns [fingerprint_size] per clip.
      Reshape to [spec_len, feat_w] → load into ring buffer → one inference.
      Produces accuracy, per-class F1, confusion-matrix PNG.

  Streaming mode  (--wav_file=clip.wav)
      Mirrors the reference _main() loop exactly:
        1. Decode WAV → raw samples
        2. Slide frame-by-frame (window_size_ms stride window_stride_ms)
        3. generate_features_tf2(): one frame → one feature row [feat_w]
        4. shift_ring_buffer() → predict() → suppression check
        5. Print per-frame detections + final result

Per-frame feature extraction (generate_features_tf2)
─────────────────────────────────────────────────────
  mfcc / average : fully TF2 eager (tf.signal)
  micro          : isolated TF1 block (_frame_features_micro_tf1)
                   same approach as input_data._preprocess_micro_tf1

Suppression
───────────
  SUPPRESSION_FRAMES  (CLI: --suppression_frames, default 25)
  After any detection, the next 25 frames are skipped.
  25 frames × 20 ms/frame = 500 ms dead-time — matches the reference.

Examples
─────────
  # Batch test-set evaluation — Keras model
  python evaluate.py \\
      --model_path=models/saved_model_tf2/model.keras \\
      --model_type=keras \\
      --data_dir=dataset/ \\
      --wanted_words=yes,no,up,down,left,right,stop,go \\
      --preprocess=mfcc

  # Batch test-set evaluation — INT8 TFLite
  python evaluate.py \\
      --model_path=models/model.tflite \\
      --model_type=tflite \\
      --data_dir=dataset/ \\
      --wanted_words=yes,no,up,down,left,right,stop,go \\
      --preprocess=mfcc \\
      --quant_input_min=-247.0 --quant_input_max=30.0

  # Streaming ring-buffer inference on a single WAV
  python evaluate.py \\
      --model_path=models/saved_model_tf2/model.keras \\
      --model_type=keras \\
      --data_dir=dataset/ \\
      --wanted_words=yes,no,up,down,left,right,stop,go \\
      --preprocess=mfcc \\
      --wav_file=test_clips/yes.wav \\
      --detection_threshold=0.7 \\
      --suppression_frames=25
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import input_data
import model_settings as model_settings_lib


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate keyword-spotting model with ring-buffer streaming.')

    # Model
    p.add_argument('--model_path', required=True,
        help='Path to a .keras file, SavedModel directory, or .tflite file.')
    p.add_argument('--model_type', required=True,
        choices=['keras', 'saved_model', 'tflite'])

    # Dataset — must match training config exactly
    p.add_argument('--data_url', default=
        'https://storage.googleapis.com/download.tensorflow.org/'
        'data/speech_commands_v0.02.tar.gz')
    p.add_argument('--data_dir',              default='dataset/')
    p.add_argument('--wanted_words',          default='yes,no,up,down,left,right,stop,go')
    p.add_argument('--silence_percentage',    type=float, default=10.0)
    p.add_argument('--unknown_percentage',    type=float, default=10.0)
    p.add_argument('--validation_percentage', type=float, default=10.0)
    p.add_argument('--testing_percentage',    type=float, default=10.0)

    # Audio / preprocessing — must match training config exactly
    p.add_argument('--sample_rate',       type=int,   default=16000)
    p.add_argument('--clip_duration_ms',  type=int,   default=1000)
    p.add_argument('--window_size_ms',    type=float, default=30.0)
    p.add_argument('--window_stride',     type=float, default=20.0,
        help='Window stride in ms — same flag name as train_tf2.py.')
    p.add_argument('--feature_bin_count', type=int,   default=40)
    p.add_argument('--preprocess',        default='mfcc',
        choices=['mfcc', 'average', 'micro'])

    # TFLite INT8 quantisation range
    p.add_argument('--quant_input_min', type=float, default=0.0,
        help='micro=0.0  mfcc=-247.0  average=0.0')
    p.add_argument('--quant_input_max', type=float, default=26.0,
        help='micro=26.0  mfcc=30.0  average=127.5')

    # Streaming mode — single WAV file
    p.add_argument('--wav_file', default='',
        help='If set, run streaming ring-buffer inference on this file. '
             'Batch test-set evaluation is skipped.')
    p.add_argument('--detection_threshold', type=float, default=0.7,
        help='Min probability to declare a keyword (reference: _DETECTION_THRESHOLD).')
    p.add_argument('--suppression_frames', type=int, default=25,
        help='Frames skipped after detection (reference: SUPPRESSION_FRAMES). '
             '25 × 20 ms = 500 ms dead-time.')

    # Output
    p.add_argument('--output_dir',  default='eval_results')
    p.add_argument('--batch_size',  type=int, default=100)

    return p.parse_args()


# ============================================================================
# Model loading
# ============================================================================

def load_keras_model(model_path):
    """Loads a .keras file or Keras SavedModel with SVDF custom objects.

    SVDFLayer and SVDFStreamingLayer are imported from train_tf2.py so that
    models trained with low_latency_svdf deserialise without error.
    """
    try:
        from train_tf2 import SVDFLayer, SVDFStreamingLayer
        custom_objects = {'SVDFLayer': SVDFLayer,
                          'SVDFStreamingLayer': SVDFStreamingLayer}
    except ImportError:
        custom_objects = {}

    model = tf.keras.models.load_model(
        model_path, custom_objects=custom_objects or None, compile=False)
    print(f'  Keras  {model.name}  '
          f'in={model.input_shape}  out={model.output_shape}')
    return model


def load_saved_model(model_path):
    """Loads a TF2 or TF1 SavedModel directory."""
    model = tf.saved_model.load(model_path)
    print(f'  SavedModel: {model_path}')
    return model


def load_tflite_model(model_path):
    """Loads a TFLite flatbuffer.  Returns (interp, inp, out, is_quantised)."""
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp  = interp.get_input_details()
    out  = interp.get_output_details()
    is_q = (inp[0]['dtype'] == np.int8)
    print(f'  TFLite  in={inp[0]["dtype"]} {inp[0]["shape"]}  '
          f'out={out[0]["dtype"]} {out[0]["shape"]}  INT8={is_q}')
    return interp, inp, out, is_q


# ============================================================================
# Ring buffer   (matches reference exactly)
# ============================================================================

# Shape mirrors _FEATURES_SHAPE = (49, 40) in the reference file.
# It is derived from model_settings at runtime rather than hard-coded so
# that non-default window sizes work correctly.

def make_ring_buffer(ms):
    """Returns a zero float32 ring buffer of shape [spectrogram_length, fingerprint_width].

    Equivalent to:
        ring_buffer = np.zeros(_FEATURES_SHAPE, dtype=dtype)
    in the reference file.  dtype is always float32 here because
    quantisation is handled inside predict() / infer_tflite().

    Args:
        ms: model_settings dict.
    """
    return np.zeros(
        (ms['spectrogram_length'], ms['fingerprint_width']),
        dtype=np.float32)


def shift_ring_buffer(ring_buffer, new_feature):
    """Shifts ring buffer forward by one row, appending new_feature at the end.

    Direct copy of reference:
        buffer[:-1] = buffer[1:]
        buffer[-1]  = new_feature

    Args:
        ring_buffer: float32 ndarray [spectrogram_length, fingerprint_width].
        new_feature: float32 ndarray [fingerprint_width] — newest frame.
    """
    ring_buffer[:-1] = ring_buffer[1:]
    ring_buffer[-1]  = new_feature


# ============================================================================
# Per-frame feature extraction   (reference: generate_features / audio_pp)
# ============================================================================

def _frame_features_micro_eager(frame_samples, ms):
    """Applies audio_microfrontend eagerly to one audio window.

    No TF1 session or placeholder.  The op is called directly as an
    EagerTensor operation, matching the reference file's op signature
    (tensorflow.lite.experimental.microfrontend.ops.audio_microfrontend_op)
    exactly.

    Parameters match the reference audio_microfrontend() signature.
    out_type=tf.float32 + 10.0/256.0 scaling is kept for consistency
    with the training pipeline.

    Args:
        frame_samples: float32 ndarray [window_size_samples], range [-1, 1].
        ms:            model_settings dict.

    Returns:
        float32 ndarray [fingerprint_width] — one ring-buffer row.
    """
    try:
        from tensorflow.lite.experimental.microfrontend.python.ops \
            import audio_microfrontend_op as frontend_op
    except ImportError:
        raise RuntimeError(
            'preprocess=micro requires the audio_microfrontend C++ op '
            '(tensorflow.lite.experimental.microfrontend). '
            'Use --preprocess=mfcc or --preprocess=average instead.')

    sample_rate    = ms['sample_rate']
    window_size_ms = ms['window_size_samples'] * 1000 / sample_rate
    window_step_ms = ms['window_stride_samples'] * 1000 / sample_rate

    int16 = tf.cast(tf.multiply(tf.constant(frame_samples), 32768), tf.int16)

    mfe = frontend_op.audio_microfrontend(
        int16,
        sample_rate=sample_rate,
        window_size=window_size_ms,
        window_step=window_step_ms,
        num_channels=ms['fingerprint_width'],
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

    # squeeze [1, fingerprint_width] → [fingerprint_width]
    return tf.multiply(mfe, 10.0 / 256.0).numpy().squeeze(axis=0).astype(np.float32)


def generate_features_tf2(frame_samples, ms):
    """Extracts one feature row from a single audio window.

    This is the TF2-native equivalent of the reference generate_features()
    / audio_preprocessor.generate_feature_using_tflm().  It produces one
    row of the ring buffer ([fingerprint_width] values) from one audio
    window of length window_size_samples.

    mfcc:    tf.signal.stft → mel filterbank → log → DCT → row[0]
    average: tf.signal.stft → magnitude → avg_pool → row[0]
    micro:   _frame_features_micro_tf1() (isolated TF1 block)

    Args:
        frame_samples: float32 ndarray [window_size_samples], range [-1, 1].
        ms:            model_settings dict.

    Returns:
        float32 ndarray [fingerprint_width] — one row for the ring buffer.
    """
    preprocess = ms['preprocess']

    if preprocess == 'micro':
        return _frame_features_micro_eager(frame_samples, ms)

    # Run STFT on one window.  frame_step = frame_length so we get exactly
    # one output frame ([1, F_bins]).
    wav_t     = tf.constant(frame_samples, dtype=tf.float32)
    stft      = tf.signal.stft(
        wav_t,
        frame_length=ms['window_size_samples'],
        frame_step=ms['window_size_samples'],   # single frame → T=1
        pad_end=True)
    magnitude = tf.abs(stft)    # [1, F_bins]

    if preprocess == 'mfcc':
        sr           = ms['sample_rate']
        num_mel      = 128
        num_mfccs    = ms['fingerprint_width']
        l2m = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel,
            num_spectrogram_bins=magnitude.shape[-1],
            sample_rate=sr,
            lower_edge_hertz=20.0,
            upper_edge_hertz=sr / 2.0)          # [F_bins, num_mel]
        mel     = tf.tensordot(magnitude, l2m, 1)   # [1, num_mel]
        log_mel = tf.math.log(mel + 1e-6)
        mfcc    = tf.signal.dcts(log_mel, n=num_mfccs, type=2,
                                 axis=-1)[..., :num_mfccs]  # [1, num_mfccs]
        return mfcc.numpy()[0].astype(np.float32)   # [fingerprint_width]

    if preprocess == 'average':
        avg_w  = ms['average_window_width']
        exp    = tf.reshape(magnitude, [1, 1, tf.shape(magnitude)[-1], 1])
        pooled = tf.nn.avg_pool2d(
            exp,
            ksize=[1, 1, avg_w, 1],
            strides=[1, 1, avg_w, 1],
            padding='SAME')                     # [1, 1, fingerprint_width, 1]
        return pooled.numpy().flatten()[:ms['fingerprint_width']].astype(np.float32)

    raise ValueError('Unknown preprocess mode "%s"' % preprocess)


# ============================================================================
# predict()   (mirrors reference predict() for all three backends)
# ============================================================================

def predict(model_type, model_obj, ring_buffer, ms,
            input_scale=None, input_zero_point=None):
    """Runs one inference on the full ring buffer.

    Mirrors the reference predict(interpreter, features) but supports
    all three model backends (keras / saved_model / tflite) and handles
    quantisation internally.

    Input to the model is always the flattened ring buffer:
        ring_buffer.flatten().reshape([1, -1])   shape [1, fingerprint_size]
    This is identical to:
        flattened = features.flatten().reshape([1, -1])   in the reference.

    Args:
        model_type:       'keras', 'saved_model', or 'tflite'.
        model_obj:        Loaded model (type depends on model_type).
        ring_buffer:      float32 ndarray [spectrogram_length, fingerprint_width].
        ms:               model_settings dict.
        input_scale:      TFLite quantisation scale   (tflite only).
        input_zero_point: TFLite quantisation zero-point (tflite only).

    Returns:
        float32 probabilities ndarray [label_count].
    """
    # Flatten ring buffer → [1, fingerprint_size]  (reference: flattened = features.flatten().reshape([1, -1]))
    features = ring_buffer.flatten().reshape(1, ms['fingerprint_size']).astype(np.float32)

    if model_type == 'keras':
        logits = model_obj(features, training=False)
        return tf.nn.softmax(logits).numpy()[0]

    if model_type == 'saved_model':
        x = tf.constant(features)
        try:
            infer  = model_obj.signatures['serving_default']
            result = infer(x)
            logits = result[list(result.keys())[0]].numpy()
        except (AttributeError, KeyError):
            logits = model_obj(x, training=False).numpy()
        return tf.nn.softmax(logits).numpy()[0]

    # tflite
    interp, inp_det, out_det, is_quantised = model_obj

    sample = features.copy()
    if is_quantised:
        # reference: quantize_input_data()
        # data = data / scale + zero_point
        if input_scale and input_scale > 0:
            sample = sample / input_scale + input_zero_point
        sample = np.clip(sample, -128, 127).astype(np.int8)

    interp.set_tensor(inp_det[0]['index'], sample)
    interp.invoke()
    output = interp.get_tensor(out_det[0]['index'])

    if out_det[0]['dtype'] == np.int8:
        # reference: dequantize_output_data()
        # scale * (data - zero_point)
        out_scale, out_zp = out_det[0]['quantization']
        output = out_scale * (output.astype(np.float32) - out_zp)

    return tf.nn.softmax(output).numpy()[0]


# ============================================================================
# Streaming inference on a single WAV file
# (mirrors reference _main() loop)
# ============================================================================

def infer_wav_streaming(wav_path, model_type, model_obj, ms, words_list, args):
    """Runs the streaming ring-buffer inference loop on one WAV file.

    Mirrors the reference _main() loop structure exactly:

        ring_buffer = np.zeros(_FEATURES_SHAPE, dtype)
        for feature in features:                    ← generate_features_tf2()
            shift_ring_buffer(ring_buffer, feature)
            probabilities = predict(interpreter, ring_buffer)
            ...suppression logic...

    Args:
        wav_path:   Path to the .wav file.
        model_type: 'keras', 'saved_model', or 'tflite'.
        model_obj:  Loaded model.
        ms:         model_settings dict.
        words_list: Ordered label list from prepare_words_list().
        args:       Parsed CLI args.
    """
    # ── Load audio ────────────────────────────────────────────────────────
    raw          = tf.io.read_file(wav_path)
    audio, sr    = tf.audio.decode_wav(raw, desired_channels=1)
    samples      = audio.numpy().flatten()          # [N] float32 in [-1, 1]
    sample_rate  = int(sr.numpy())

    window_size   = ms['window_size_samples']       # 480
    window_stride = ms['window_stride_samples']     # 320
    spec_len      = ms['spectrogram_length']        # 49  = _FEATURES_SHAPE[0]

    print(f'\n[streaming]  {wav_path}')
    print(f'  {len(samples)} samples  ({len(samples)/sample_rate*1000:.0f} ms)'
          f'  @{sample_rate} Hz')
    print(f'  ring_buffer shape    = ({spec_len}, {ms["fingerprint_width"]})'
          f'  — _FEATURES_SHAPE = (49, 40)')
    print(f'  detection_threshold  = {args.detection_threshold}')
    print(f'  suppression_frames   = {args.suppression_frames}'
          f'  ({args.suppression_frames * args.window_stride:.0f} ms)')

    # ── TFLite quant params ────────────────────────────────────────────────
    input_scale = input_zero_point = None
    if model_type == 'tflite':
        _, inp_det, _, _ = model_obj
        input_scale      = inp_det[0]['quantization'][0]
        input_zero_point = inp_det[0]['quantization'][1]

    # ── generate_features_tf2()  ──────────────────────────────────────────
    # Reference: features = generate_features(audio_pp)  → [49, 40]
    # Here we generate each frame on the fly inside the loop so memory
    # usage stays constant (no pre-allocation of the full feature matrix).

    # ── ring_buffer = np.zeros(_FEATURES_SHAPE, dtype)  (reference line) ──
    ring_buffer         = make_ring_buffer(ms)

    frame_number        = 0
    suppression_counter = 0
    detected_label      = words_list[0]     # 'silence'
    detected_prob       = 0.0
    detections          = []

    print('\nStreaming inference started')

    start = 0
    while start + window_size <= len(samples) and frame_number < spec_len * 4:
        frame = samples[start: start + window_size]

        # generate one feature row  (reference: feature_tensor = audio_pp.generate_feature_using_tflm())
        feature = generate_features_tf2(frame, ms)          # [fingerprint_width]

        # reference: shift_ring_buffer(ring_buffer, feature)
        shift_ring_buffer(ring_buffer, feature)

        # reference: probabilities = predict(interpreter, ring_buffer)
        probabilities   = predict(model_type, model_obj, ring_buffer, ms,
                                  input_scale, input_zero_point)

        predicted_index = int(np.argmax(probabilities))
        probability     = float(probabilities[predicted_index])
        label           = words_list[predicted_index]

        # reference: suppression logic
        if suppression_counter > 0:
            suppression_counter -= 1
            frame_number        += 1
            start               += window_stride
            continue

        # reference: if predicted_index >= 2 and probability >= _DETECTION_THRESHOLD
        if predicted_index >= 2 and probability >= args.detection_threshold:
            ts_ms = frame_number * args.window_stride
            print(f"  Detected '{label}' "
                  f"(prob={probability:.2f}) "
                  f"at frame {frame_number} ({ts_ms:.0f} ms)")
            detections.append((frame_number, ts_ms, label, probability))
            detected_label      = label
            detected_prob       = probability
            suppression_counter = args.suppression_frames

        frame_number += 1
        start        += window_stride

    # reference: print("Final prediction:", detected_label, ...)
    print(f'\n  frames processed : {frame_number}')
    print(f'  detections       : {len(detections)}')
    print(f'Final prediction: {detected_label} (prob={detected_prob:.2f})')

    return detected_label, detected_prob, detections


# ============================================================================
# Batch test-set evaluation via ring buffer
# ============================================================================

def evaluate_test_set(model_type, model_obj, audio_proc, ms, words_list, args):
    """Evaluates the full test partition; each clip is loaded into the ring buffer.

    AudioProcessor.get_data() returns flat fingerprints [fingerprint_size].
    Each is reshaped to [spectrogram_length, fingerprint_width] and loaded
    directly into the ring buffer (warm-start: all 49 rows populated at once),
    then predict() is called once.  This exercises the identical ring-buffer →
    model path used by streaming mode, giving comparable accuracy numbers.

    Args:
        model_type:  'keras', 'saved_model', or 'tflite'.
        model_obj:   Loaded model.
        audio_proc:  Initialised AudioProcessor.
        ms:          model_settings dict.
        words_list:  Ordered label list.
        args:        Parsed CLI args.

    Returns:
        (y_true, y_pred) integer numpy arrays, shape [N].
    """
    spec_len   = ms['spectrogram_length']   # 49
    feat_width = ms['fingerprint_width']    # 40
    set_size   = audio_proc.set_size('testing')
    y_true, y_pred = [], []

    input_scale = input_zero_point = None
    if model_type == 'tflite':
        _, inp_det, _, _ = model_obj
        input_scale      = inp_det[0]['quantization'][0]
        input_zero_point = inp_det[0]['quantization'][1]

    print(f'\nEvaluating {set_size} test samples '
          f'(ring buffer [{spec_len}×{feat_width}]) …')
    processed = 0

    for offset in range(0, set_size, args.batch_size):
        fingerprints, labels = audio_proc.get_data(
            how_many=args.batch_size,
            offset=offset,
            model_settings=ms,
            background_frequency=0.0,
            background_volume_range=0.0,
            time_shift=0,
            mode='testing')

        for i in range(len(fingerprints)):
            # Reshape flat fingerprint → ring buffer shape [49, 40]
            ring_buffer = fingerprints[i].reshape(
                spec_len, feat_width).astype(np.float32)

            probs = predict(model_type, model_obj, ring_buffer, ms,
                            input_scale, input_zero_point)

            y_true.append(int(labels[i]))
            y_pred.append(int(np.argmax(probs)))

        processed += len(fingerprints)
        if processed % 500 == 0 or processed >= set_size:
            print(f'  {processed}/{set_size}', end='\r')

    print()
    return np.array(y_true), np.array(y_pred)


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(y_true, y_pred, words_list):
    n  = len(words_list)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total   = len(y_true)
    correct = int(np.sum(y_true == y_pred))

    per_class = []
    for i, word in enumerate(words_list):
        tp   = cm[i, i]
        fp   = cm[:, i].sum() - tp
        fn   = cm[i, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class.append({'word': word, 'precision': prec,
                          'recall': rec, 'f1': f1,
                          'support': int(cm[i, :].sum())})

    return {'accuracy': 100.0 * correct / max(total, 1),
            'correct': correct, 'total': total,
            'per_class': per_class, 'confusion_matrix': cm}


def print_metrics(metrics, words_list):
    print(f'\n{"─"*62}')
    print(f'  Overall accuracy : {metrics["accuracy"]:.2f}%  '
          f'({metrics["correct"]}/{metrics["total"]})')
    print(f'{"─"*62}')
    print(f'  {"Label":<22}  {"Prec":>6}  {"Recall":>6}  {"F1":>6}  {"N":>6}')
    print(f'  {"─"*22}  {"─"*6}  {"─"*6}  {"─"*6}  {"─"*6}')
    for pc in metrics['per_class']:
        flag = '  ← low' if pc['f1'] < 0.70 and pc['support'] > 0 else ''
        print(f'  {pc["word"]:<22}  {pc["precision"]:>6.3f}  '
              f'{pc["recall"]:>6.3f}  {pc["f1"]:>6.3f}  '
              f'{pc["support"]:>6}{flag}')
    print(f'{"─"*62}')

    cm   = metrics['confusion_matrix']
    conf = [(cm[i, j], words_list[i], words_list[j])
            for i in range(len(words_list))
            for j in range(len(words_list))
            if i != j and cm[i, j] > 0]
    conf.sort(reverse=True)
    if conf:
        print('\n  Most confused pairs (true → predicted):')
        for count, tw, pw in conf[:5]:
            print(f'    {tw:>22} → {pw:<22}  ({count}×)')
    print()


def save_confusion_matrix(cm, words_list, output_path):
    fig, ax = plt.subplots(figsize=(max(8, len(words_list)),
                                    max(6, len(words_list) - 2)))
    sns.heatmap(cm, xticklabels=words_list, yticklabels=words_list,
                annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5, linecolor='lightgrey')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'  confusion matrix : {output_path}')


def save_metrics_text(metrics, words_list, output_path):
    lines = [
        f'Overall accuracy: {metrics["accuracy"]:.4f}%  '
        f'({metrics["correct"]}/{metrics["total"]})\n',
        f'{"Label":<22}  {"Precision":>9}  {"Recall":>9}  '
        f'{"F1":>9}  {"Support":>9}\n',
        '─' * 62 + '\n',
    ]
    for pc in metrics['per_class']:
        lines.append(f'{pc["word"]:<22}  {pc["precision"]:>9.4f}  '
                     f'{pc["recall"]:>9.4f}  {pc["f1"]:>9.4f}  '
                     f'{pc["support"]:>9}\n')
    with open(output_path, 'w') as f:
        f.writelines(lines)
    print(f'  metrics text     : {output_path}')


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model settings  (same function as train_tf2.py) ───────────────────
    wanted_words_list = args.wanted_words.split(',')
    words_list        = input_data.prepare_words_list(wanted_words_list)

    ms = model_settings_lib.prepare_model_settings(
        label_count       = len(words_list),
        sample_rate       = args.sample_rate,
        clip_duration_ms  = args.clip_duration_ms,
        window_size_ms    = args.window_size_ms,
        window_stride_ms  = args.window_stride,
        feature_bin_count = args.feature_bin_count,
        preprocess        = args.preprocess,
    )
    print(f'ring_buffer=[{ms["spectrogram_length"]}×{ms["fingerprint_width"]}]  '
          f'fingerprint_size={ms["fingerprint_size"]}  '
          f'labels={len(words_list)}  preprocess={args.preprocess}')

    # ── Load model ─────────────────────────────────────────────────────────
    print(f'\nLoading {args.model_type}: {args.model_path}')
    if args.model_type == 'keras':
        model_obj = load_keras_model(args.model_path)
    elif args.model_type == 'saved_model':
        model_obj = load_saved_model(args.model_path)
    else:
        model_obj = load_tflite_model(args.model_path)

    # ── Streaming mode ─────────────────────────────────────────────────────
    if args.wav_file:
        infer_wav_streaming(args.wav_file, args.model_type, model_obj,
                            ms, words_list, args)
        return

    # ── Batch test-set mode ────────────────────────────────────────────────
    print('\nInitialising AudioProcessor …')
    audio_proc = input_data.AudioProcessor(
        data_url              = args.data_url,
        data_dir              = args.data_dir,
        silence_percentage    = args.silence_percentage,
        unknown_percentage    = args.unknown_percentage,
        wanted_words          = wanted_words_list,
        validation_percentage = args.validation_percentage,
        testing_percentage    = args.testing_percentage,
        model_settings        = ms,
        summaries_dir         = '',
    )

    y_true, y_pred = evaluate_test_set(
        args.model_type, model_obj, audio_proc, ms, words_list, args)

    metrics = compute_metrics(y_true, y_pred, words_list)
    print_metrics(metrics, words_list)

    save_confusion_matrix(
        metrics['confusion_matrix'], words_list,
        os.path.join(args.output_dir, 'confusion_matrix.png'))
    save_metrics_text(
        metrics, words_list,
        os.path.join(args.output_dir, 'metrics.txt'))

    print(f'\nAll results written to: {args.output_dir}/')


if __name__ == '__main__':
    main()