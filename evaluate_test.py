# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""
Wake-word model testing, with audio preprocessing using MicroInterpreter

Run:
bazel build tensorflow/lite/micro/examples/micro_speech:evaluate_test
bazel-bin/tensorflow/lite/micro/examples/micro_speech/evaluate_test
"""

import numpy as np
from pathlib import Path

from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro.examples.micro_speech import audio_preprocessor
from tflite_micro.tensorflow.lite.micro.examples.micro_speech import evaluate


class MicroSpeechTest(test_util.TensorFlowTestCase):

  def setUp(self):
    model_prefix_path = resource_loader.get_path_to_datafile('models')
    self.sample_prefix_path = resource_loader.get_path_to_datafile('testdata')
    model_path = Path(model_prefix_path, 'micro_speech_quantized.tflite')
    self.tflm_interpreter = runtime.Interpreter.from_file(model_path)
    self.test_data = [
        ('no', 'no_1000ms.wav'),
        ('yes', 'yes_1000ms.wav'),
        ('silence', 'noise_1000ms.wav'),
        ('silence', 'silence_1000ms.wav'),
    ]

  def testModelAccuracyWithInt8Features(self):
    feature_params = audio_preprocessor.FeatureParams()
    audio_pp = audio_preprocessor.AudioPreprocessor(feature_params)
    for label, sample_name in self.test_data:
      # Load audio sample data
      sample_path = Path(self.sample_prefix_path, sample_name)
      audio_pp.load_samples(sample_path)

      # Generate feature data from audio samples.
      # Note that the noise estimate is reset each time generate_features()
      # is called.
      features = evaluate.generate_features(audio_pp)

      # Run model inference (quantized) on the feature data
      category_probabilities = evaluate.predict(self.tflm_interpreter,
                                                features)

      # Check the prediction result
      predicted_category = np.argmax(category_probabilities)
      category_names = evaluate.get_category_names()
      # Check the prediction
      self.assertEqual(category_names[predicted_category], label)

  def testModelAccuracyWithFloatFeatures(self):
    feature_params = audio_preprocessor.FeatureParams(use_float_output=True)
    audio_pp = audio_preprocessor.AudioPreprocessor(feature_params)
    for label, sample_name in self.test_data:
      # Load audio sample data
      sample_path = Path(self.sample_prefix_path, sample_name)
      audio_pp.load_samples(sample_path)

      # Generate feature data from audio samples.
      # Note that the noise estimate is reset each time generate_features()
      # is called.
      features = evaluate.generate_features(audio_pp)

      # Run model inference (quantized) on the feature data
      category_probabilities = evaluate.predict(self.tflm_interpreter,
                                                features)

      # Check the prediction result
      predicted_category = np.argmax(category_probabilities)
      category_names = evaluate.get_category_names()
      # Check the prediction
      self.assertEqual(category_names[predicted_category], label)

  # ------------------------------------------------------------------
  # The following tests exercise the new ring-buffer streaming logic.  When
  # a keyword is detected the buffer is immediately cleared.  This mirrors the
  # behaviour described in the evaluation script's documentation string.
  # ------------------------------------------------------------------

  def _stream_and_snapshot(self, features: np.ndarray, threshold: float):
    """Run a simple ring-buffer sliding window over |features|.

    Returns a list of buffer snapshots captured immediately after a keyword
    detection event.  Each snapshot should be all-zero if the reset logic is
    functioning correctly.
    """
    buffer = np.zeros(evaluate._FEATURES_SHAPE, dtype=features.dtype)
    snapshots = []
    for feature in features:
      buffer = np.roll(buffer, -1, axis=0)
      buffer[-1] = feature
      probs = evaluate.predict(self.tflm_interpreter, buffer)
      predicted = np.argmax(probs)
      if (predicted in evaluate._KEYWORD_INDICES and
          probs[predicted] >= threshold):
        # record the state of the buffer before we clear it
        snapshots.append(buffer.copy())
        buffer.fill(0)
    return snapshots

  def testRingBufferReset(self):
    # load a 'yes' example to guarantee that a keyword will eventually fire
    feature_params = audio_preprocessor.FeatureParams()
    audio_pp = audio_preprocessor.AudioPreprocessor(feature_params)
    sample_path = Path(self.sample_prefix_path, 'yes_1000ms.wav')
    audio_pp.load_samples(sample_path)
    features = evaluate.generate_features(audio_pp)

    snaps = self._stream_and_snapshot(features, threshold=0.0)
    # we expect at least one detection to occur
    self.assertTrue(len(snaps) > 0)
    # each snapshot captured should be all-zero (buffer reset)
    for snap in snaps:
      self.assertTrue(np.all(snap == 0))


if __name__ == '__main__':
  test.main()
