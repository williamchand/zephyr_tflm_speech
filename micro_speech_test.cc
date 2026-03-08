/* Copyright 2024 The TensorFlow Authors.
Licensed under the Apache License, Version 2.0 */

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/noise_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/silence_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms_audio_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

constexpr size_t kArenaSize = 28584;
alignas(16) uint8_t g_arena[kArenaSize];

using Features = int8_t[kFeatureCount][kFeatureSize];
Features g_features;

constexpr int kAudioSampleDurationCount =
    kFeatureDurationMs * kAudioSampleFrequency / 1000;
constexpr int kAudioSampleStrideCount =
    kFeatureStrideMs * kAudioSampleFrequency / 1000;

// suppression window for streaming detection
constexpr int kSuppressionFrames = 25;

using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<4>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

TfLiteStatus RegisterOps(MicroSpeechOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

TfLiteStatus RegisterOps(AudioPreprocessorOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
  return kTfLiteOk;
}

TfLiteStatus LoadMicroSpeechModelAndPerformInference(
    const Features& features, const char* expected_label) {

  const tflite::Model* model =
      tflite::GetModel(g_micro_speech_quantized_model_data);

  MicroSpeechOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  std::copy_n(&features[0][0], kFeatureElementCount,
              tflite::GetTensorData<int8_t>(input));

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  TfLiteTensor* output = interpreter.output(0);

  float scale = output->params.scale;
  int zero_point = output->params.zero_point;

  float best_score = -1e9f;
  int best_index = 0;

  for (int i = 0; i < kCategoryCount; ++i) {
    float score =
        (tflite::GetTensorData<int8_t>(output)[i] - zero_point) * scale;

    if (score > best_score) {
      best_score = score;
      best_index = i;
    }
  }

  if (strcmp(expected_label, kCategoryLabels[best_index]) != 0) {
    MicroPrintf("Expected label mismatch!");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output,
                                   tflite::MicroInterpreter* interpreter) {

  TfLiteTensor* input = interpreter->input(0);
  std::copy_n(audio_data, audio_data_size,
              tflite::GetTensorData<int16_t>(input));

  TF_LITE_ENSURE_STATUS(interpreter->Invoke());

  std::copy_n(tflite::GetTensorData<int8_t>(interpreter->output(0)),
              kFeatureSize, feature_output);

  return kTfLiteOk;
}

TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output) {

  const tflite::Model* model =
      tflite::GetModel(g_audio_preprocessor_int8_model_data);

  AudioPreprocessorOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  size_t remaining_samples = audio_data_size;
  size_t feature_index = 0;

  while (remaining_samples >= kAudioSampleDurationCount &&
         feature_index < kFeatureCount) {

    TF_LITE_ENSURE_STATUS(
        GenerateSingleFeature(audio_data,
                              kAudioSampleDurationCount,
                              (*features_output)[feature_index],
                              &interpreter));

    feature_index++;
    audio_data += kAudioSampleStrideCount;
    remaining_samples -= kAudioSampleStrideCount;
  }

  return kTfLiteOk;
}

TfLiteStatus TestAudioSample(const char* label,
                             const int16_t* audio_data,
                             const size_t audio_data_size) {

  TF_LITE_ENSURE_STATUS(
      GenerateFeatures(audio_data, audio_data_size, &g_features));

  TF_LITE_ENSURE_STATUS(
      LoadMicroSpeechModelAndPerformInference(g_features, label));

  return kTfLiteOk;
}

// ring buffer shift
void ShiftRingBuffer(Features& buffer, const int8_t* new_feature) {
  for (int r = 0; r < kFeatureCount - 1; ++r) {
    std::copy(&buffer[r + 1][0],
              &buffer[r + 1][0] + kFeatureSize,
              &buffer[r][0]);
  }

  std::copy(new_feature,
            new_feature + kFeatureSize,
            &buffer[kFeatureCount - 1][0]);
}

int PredictCategory(const Features& features) {

  const tflite::Model* model =
      tflite::GetModel(g_micro_speech_quantized_model_data);

  MicroSpeechOpResolver op_resolver;
  RegisterOps(op_resolver);

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);
  interpreter.AllocateTensors();

  std::copy_n(&features[0][0], kFeatureElementCount,
              tflite::GetTensorData<int8_t>(interpreter.input(0)));

  interpreter.Invoke();

  TfLiteTensor* output = interpreter.output(0);

  float scale = output->params.scale;
  int zero_point = output->params.zero_point;

  int best_index = 0;
  float best_score = -1e9f;

  for (int i = 0; i < kCategoryCount; ++i) {

    float score =
        (tflite::GetTensorData<int8_t>(output)[i] - zero_point) * scale;

    if (score > best_score) {
      best_score = score;
      best_index = i;
    }
  }

  return best_index;
}

}  // namespace


TEST(MicroSpeechTest, NoTest) {
  ASSERT_EQ(TestAudioSample("no",
                            g_no_1000ms_audio_data,
                            g_no_1000ms_audio_data_size),
            kTfLiteOk);
}

TEST(MicroSpeechTest, YesTest) {
  ASSERT_EQ(TestAudioSample("yes",
                            g_yes_1000ms_audio_data,
                            g_yes_1000ms_audio_data_size),
            kTfLiteOk);
}

TEST(MicroSpeechTest, SilenceTest) {
  ASSERT_EQ(TestAudioSample("silence",
                            g_silence_1000ms_audio_data,
                            g_silence_1000ms_audio_data_size),
            kTfLiteOk);
}

TEST(MicroSpeechTest, NoiseTest) {
  ASSERT_EQ(TestAudioSample("silence",
                            g_noise_1000ms_audio_data,
                            g_noise_1000ms_audio_data_size),
            kTfLiteOk);
}

// streaming test with suppression
TEST(MicroSpeechTest, RingBufferSuppression) {

  Features yes_features;

  ASSERT_EQ(
      GenerateFeatures(g_yes_1000ms_audio_data,
                       g_yes_1000ms_audio_data_size,
                       &yes_features),
      kTfLiteOk);

  Features ring = {};

  bool saw_keyword = false;
  int suppression_counter = 0;

  for (int frame = 0; frame < kFeatureCount; ++frame) {

    ShiftRingBuffer(ring, yes_features[frame]);

    if (suppression_counter > 0) {
      suppression_counter--;
      continue;
    }

    int predicted = PredictCategory(ring);

    if (predicted >= 2) {
      saw_keyword = true;
      suppression_counter = kSuppressionFrames;
    }
  }

  EXPECT_TRUE(saw_keyword);
}

TF_LITE_MICRO_TESTS_MAIN