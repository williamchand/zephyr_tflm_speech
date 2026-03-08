# Copyright 2023 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0

from absl import app
from absl import flags
import numpy as np
from pathlib import Path

from tflite_micro.python.tflite_micro import runtime
from tensorflow.python.platform import resource_loader
import tensorflow as tf
from tflite_micro.tensorflow.lite.micro.examples.micro_speech import audio_preprocessor


_SAMPLE_PATH = flags.DEFINE_string(
    name='sample_path',
    default='',
    help='path for the audio sample to be predicted.',
)

_FEATURES_SHAPE = (49, 40)

_DETECTION_THRESHOLD = flags.DEFINE_float(
    'detection_threshold', 0.7,
    'Probability threshold used for keyword detection.'
)

CATEGORY_NAMES = [
    "silence",
    "unknown",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "stop",
    "go"
]


def quantize_input_data(data, input_details):

  data_type = input_details['dtype']
  params = input_details['quantization_parameters']

  scale = params['scales'][0]
  zero_point = params['zero_points'][0]

  data = data / scale + zero_point
  return data.astype(data_type)


def dequantize_output_data(data, output_details):

  params = output_details['quantization_parameters']

  scale = params['scales'][0]
  zero_point = params['zero_points'][0]

  return scale * (data.astype(np.float32) - zero_point)


def predict(interpreter, features):

  input_details = interpreter.get_input_details(0)

  if input_details['dtype'] != np.float32 and features.dtype == np.float32:
    features = quantize_input_data(features, input_details)

  flattened = features.flatten().reshape([1, -1])

  interpreter.set_input(flattened, 0)
  interpreter.invoke()

  output = interpreter.get_output(0)

  output_details = interpreter.get_output_details(0)

  if output_details['dtype'] == np.float32:
    return output[0].astype(np.float32)

  return dequantize_output_data(output[0], output_details)


def generate_features(audio_pp):

  if audio_pp.params.use_float_output:
    dtype = np.float32
  else:
    dtype = np.int8

  features = np.zeros(_FEATURES_SHAPE, dtype=dtype)

  start_index = 0

  window_size = int(
      audio_pp.params.window_size_ms *
      audio_pp.params.sample_rate / 1000)

  window_stride = int(
      audio_pp.params.window_stride_ms *
      audio_pp.params.sample_rate / 1000)

  samples = audio_pp.samples[0]

  frame_number = 0
  end_index = start_index + window_size

  audio_pp.reset_tflm()

  while end_index <= len(samples) and frame_number < _FEATURES_SHAPE[0]:

    frame_tensor = tf.convert_to_tensor(samples[start_index:end_index])
    frame_tensor = tf.reshape(frame_tensor, [1, -1])

    feature_tensor = audio_pp.generate_feature_using_tflm(frame_tensor)

    features[frame_number] = feature_tensor.numpy()

    start_index += window_stride
    end_index += window_stride

    frame_number += 1

  return features


def shift_ring_buffer(buffer, new_feature):
  buffer[:-1] = buffer[1:]
  buffer[-1] = new_feature


def zero_ring_buffer(buffer):
  buffer[:] = 0


def _main(_):

  sample_path = Path(_SAMPLE_PATH.value)

  assert sample_path.exists() and sample_path.is_file(), \
      "Audio sample file does not exist."

  model_prefix_path = resource_loader.get_path_to_datafile('models')
  model_path = Path(model_prefix_path, 'micro_speech_quantized.tflite')

  feature_params = audio_preprocessor.FeatureParams()
  audio_pp = audio_preprocessor.AudioPreprocessor(feature_params)

  audio_pp.load_samples(sample_path)

  features = generate_features(audio_pp)

  interpreter = runtime.Interpreter.from_file(model_path)

  dtype = np.float32 if audio_pp.params.use_float_output else np.int8

  ring_buffer = np.zeros(_FEATURES_SHAPE, dtype=dtype)

  frame_number = 0
  detected_label = "silence"
  detected_prob = 0.0

  print("Streaming inference started")

  for feature in features:

    # shift buffer and append new frame
    shift_ring_buffer(ring_buffer, feature)

    probabilities = predict(interpreter, ring_buffer)

    predicted_index = np.argmax(probabilities)
    probability = probabilities[predicted_index]

    label = CATEGORY_NAMES[predicted_index]

    if predicted_index >= 2 and probability >= _DETECTION_THRESHOLD.value:

      print(
          f"Detected '{label}' "
          f"(prob={probability:.2f}) "
          f"at frame {frame_number}"
      )

      detected_label = label
      detected_prob = probability

      zero_ring_buffer(ring_buffer)

      break

    frame_number += 1

  print(
      "Final prediction:",
      detected_label,
      f"(prob={detected_prob:.2f})"
  )


if __name__ == '__main__':
  app.run(_main)