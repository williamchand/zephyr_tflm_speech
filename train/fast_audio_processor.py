import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.platform import gfile

import input_data


class FastAudioProcessor(input_data.AudioProcessor):
    """
    TF2 optimized version of AudioProcessor.

    Improvements:
    • removes per-sample sess.run
    • vectorized audio processing
    • tf.data pipeline
    • parallel CPU feature extraction
    • GPU friendly
    """

    def __init__(
        self,
        data_url,
        data_dir,
        silence_percentage,
        unknown_percentage,
        wanted_words,
        validation_percentage,
        testing_percentage,
        model_settings,
        summaries_dir,
    ):

        super().__init__(
            data_url,
            data_dir,
            silence_percentage,
            unknown_percentage,
            wanted_words,
            validation_percentage,
            testing_percentage,
            model_settings,
            summaries_dir,
        )

        self.model_settings = model_settings
        self.desired_samples = model_settings["desired_samples"]

    # --------------------------------------------------------
    # TF2 AUDIO PIPELINE
    # --------------------------------------------------------

    def _load_wav(self, filename):

        audio_binary = tf.io.read_file(filename)

        wav, _ = tf.audio.decode_wav(
            audio_binary,
            desired_channels=1,
            desired_samples=self.desired_samples,
        )

        return wav

    def _augment(self, wav, label, background_frequency,
                 background_volume_range, time_shift):

        # time shift
        if time_shift > 0:

            shift = tf.random.uniform(
                [],
                -time_shift,
                time_shift,
                dtype=tf.int32,
            )

            wav = tf.roll(wav, shift, axis=0)

        # background noise
        if self.background_data:

            if tf.random.uniform([]) < background_frequency:

                bg = random.choice(self.background_data)

                bg = bg[: self.desired_samples]

                bg = tf.convert_to_tensor(bg.reshape(-1, 1), tf.float32)

                volume = tf.random.uniform(
                    [],
                    0,
                    background_volume_range,
                )

                wav = tf.clip_by_value(wav + volume * bg, -1.0, 1.0)

        return wav, label

    def _extract_features(self, wav):

        spectrogram = audio_ops.audio_spectrogram(
            wav,
            window_size=self.model_settings["window_size_samples"],
            stride=self.model_settings["window_stride_samples"],
            magnitude_squared=True,
        )

        if self.model_settings["preprocess"] == "mfcc":

            features = audio_ops.mfcc(
                spectrogram,
                self.model_settings["sample_rate"],
                dct_coefficient_count=self.model_settings["fingerprint_width"],
            )

        elif self.model_settings["preprocess"] == "average":

            features = tf.nn.pool(
                tf.expand_dims(spectrogram, -1),
                window_shape=[1, self.model_settings["average_window_width"]],
                strides=[1, self.model_settings["average_window_width"]],
                pooling_type="AVG",
                padding="SAME",
            )

        else:

            raise ValueError("Unsupported preprocess mode")

        return tf.reshape(features, [-1])

    # --------------------------------------------------------
    # TF DATA PIPELINE
    # --------------------------------------------------------

    def dataset(
        self,
        mode,
        batch_size,
        background_frequency,
        background_volume_range,
        time_shift,
    ):

        entries = self.data_index[mode]

        filenames = [x["file"] for x in entries]
        labels = [self.word_to_index[x["label"]] for x in entries]

        ds = tf.data.Dataset.from_tensor_slices((filenames, labels))

        if mode == "training":
            ds = ds.shuffle(10000)

        def _process(file, label):

            wav = self._load_wav(file)

            wav, label = self._augment(
                wav,
                label,
                background_frequency,
                background_volume_range,
                time_shift,
            )

            features = self._extract_features(wav)

            return features, label

        ds = ds.map(
            _process,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    # --------------------------------------------------------
    # COMPATIBILITY WITH OLD train.py
    # --------------------------------------------------------

    def get_data(
        self,
        how_many,
        offset,
        model_settings,
        background_frequency,
        background_volume_range,
        time_shift,
        mode,
        sess=None,
    ):

        # fallback to TF2 dataset
        ds = self.dataset(
            mode,
            how_many,
            background_frequency,
            background_volume_range,
            time_shift,
        )

        features, labels = next(iter(ds))

        return features.numpy(), labels.numpy()