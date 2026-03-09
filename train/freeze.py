# freeze.py
import os
import argparse
import tensorflow as tf
from models import create_model, prepare_model_settings, load_variables_from_checkpoint

# ---------------------------------------------------------------------------
# Inference Module
# ---------------------------------------------------------------------------
class SpeechCommandsInferenceModel(tf.Module):
    def __init__(self, model_settings, model_architecture='conv'):
        super().__init__()
        self.model_settings = model_settings
        self.fingerprint_size = model_settings['fingerprint_size']
        self.model_architecture = model_architecture

        # Build actual Keras model
        dummy_input = tf.zeros([1, self.fingerprint_size], dtype=tf.float32)
        result = create_model(dummy_input, model_settings, model_architecture, is_training=False)
        
        # create_model may return tuple (logits, dropout)
        if isinstance(result, tuple):
            self._keras_model = result[0]._keras_model if hasattr(result[0], "_keras_model") else result[0]
        else:
            self._keras_model = result

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def infer(self, fingerprint_input):
        # Flatten input and pad/truncate to match fingerprint_size
        flattened = tf.reshape(fingerprint_input, [-1])
        size_diff = self.fingerprint_size - tf.size(flattened)
        padded = tf.cond(size_diff > 0,
                         lambda: tf.pad(flattened, [[0, size_diff]]),
                         lambda: flattened[:self.fingerprint_size])
        reshaped = tf.reshape(padded, [1, self.fingerprint_size])
        
        # Run Keras model
        logits = self._keras_model(reshaped, training=False)
        return logits

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def save_saved_model(export_dir, inference_model):
    tf.saved_model.save(
        inference_model,
        export_dir,
        signatures={'serving_default': inference_model.infer}
    )
    print(f"SavedModel exported to {export_dir} successfully!")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wanted_words', type=str, default='yes,no,up,down')
    parser.add_argument('--window_stride_ms', type=int, default=10)
    parser.add_argument('--preprocess', type=str, default='mfcc')
    parser.add_argument('--model_architecture', type=str, default='conv')
    parser.add_argument('--start_checkpoint', type=str, required=True)
    parser.add_argument('--save_format', type=str, default='saved_model')
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    # Model settings
    model_settings = prepare_model_settings(
        label_count=len(args.wanted_words.split(',')),
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=30,
        window_stride_ms=args.window_stride_ms,
        feature_bin_count=40,
        preprocess=args.preprocess
    )

    # Build inference module
    inference_module = SpeechCommandsInferenceModel(
        model_settings=model_settings,
        model_architecture=args.model_architecture
    )

    # Restore checkpoint
    print(f"Restoring checkpoint from {args.start_checkpoint} ...")
    load_variables_from_checkpoint(None, args.start_checkpoint)

    # Save as SavedModel
    if args.save_format == 'saved_model':
        save_saved_model(args.output_file, inference_module)

if __name__ == "__main__":
    main()