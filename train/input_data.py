import os
import numpy as np
import tensorflow as tf


class AudioProcessor:
    def __init__(self, data_dir, wanted_words, validation_pct, testing_pct, model_settings):
        self.data_dir = data_dir
        self.model_settings = model_settings
        self.wanted_words = wanted_words

        self.data_index = {"training": [], "validation": [], "testing": []}
        self.word_to_index = {}

        self._prepare_index(validation_pct, testing_pct)

    def _prepare_index(self, validation_pct, testing_pct):
        all_words = set()

        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if not f.endswith(".wav"):
                    continue

                label = os.path.basename(root)
                path = os.path.join(root, f)

                all_words.add(label)

                set_type = self._which_set(f, validation_pct, testing_pct)

                self.data_index[set_type].append({
                    "label": label,
                    "file": path
                })

        self.word_to_index = {w: i for i, w in enumerate(sorted(all_words))}

    def _which_set(self, filename, validation_pct, testing_pct):
        h = hash(filename) % 100
        if h < validation_pct:
            return "validation"
        elif h < validation_pct + testing_pct:
            return "testing"
        else:
            return "training"

    def _load_wav(self, path):
        audio_binary = tf.io.read_file(path)
        audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
        return tf.squeeze(audio)

    def _compute_mfcc(self, audio):
        stft = tf.signal.stft(
            audio,
            frame_length=self.model_settings["window_size_samples"],
            frame_step=self.model_settings["window_stride_samples"]
        )

        spectrogram = tf.abs(stft)

        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.model_settings["fingerprint_width"],
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.model_settings["sample_rate"]
        )

        mel_spec = tf.matmul(spectrogram, mel_matrix)

        log_mel = tf.math.log(mel_spec + 1e-6)

        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)

        return mfcc[:, :self.model_settings["fingerprint_width"]]

    def get_data(self, how_many, offset, mode):

        samples = self.data_index[mode]

        if how_many == -1:
            selected = samples
        else:
            selected = samples[offset:offset + how_many]

        data = []
        labels = []

        for s in selected:
            audio = self._load_wav(s["file"])
            features = self._compute_mfcc(audio)

            data.append(tf.reshape(features, [-1]).numpy())
            labels.append(self.word_to_index[s["label"]])

        return np.array(data), np.array(labels)