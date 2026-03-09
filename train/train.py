import argparse
import os
import time
import numpy as np
import tensorflow as tf

import models
from fast_audio_processor import FastAudioProcessor
import input_data

# -----------------------------
# Helper: TF2-style model settings
# -----------------------------
def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count, preprocess):
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    if preprocess in ("mfcc", "micro"):
        fingerprint_width = dct_coefficient_count
    elif preprocess == "average":
        fingerprint_width = dct_coefficient_count
    else:
        raise ValueError("Unknown preprocess mode")

    spectrogram_length = 1 + int((desired_samples - window_size_samples) / window_stride_samples)

    average_window_width = 10 if preprocess == "average" else None
    fingerprint_size = fingerprint_width * spectrogram_length

    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "fingerprint_width": fingerprint_width,
        "spectrogram_length": spectrogram_length,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "preprocess": preprocess,
        "average_window_width": average_window_width,
        "sample_rate": sample_rate,
    }

# -----------------------------
# Main training loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./speech_dataset")
    parser.add_argument("--wanted_words", type=str,
                        default="yes,no,up,down,left,right,on,off,stop,go")
    parser.add_argument("--validation_percentage", type=int, default=10)
    parser.add_argument("--testing_percentage", type=int, default=10)
    parser.add_argument("--silence_percentage", type=int, default=10)
    parser.add_argument("--unknown_percentage", type=int, default=10)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--clip_duration_ms", type=int, default=1000)
    parser.add_argument("--window_size_ms", type=float, default=30.0)
    parser.add_argument("--window_stride_ms", type=float, default=10.0)
    parser.add_argument("--dct_coefficient_count", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--how_many_training_steps", type=str, default="20000,10000")
    parser.add_argument("--learning_rate", type=str, default="0.001,0.0001")
    parser.add_argument("--model_architecture", type=str, default="conv")
    parser.add_argument("--background_frequency", type=float, default=0.8)
    parser.add_argument("--background_volume", type=float, default=0.1)
    parser.add_argument("--time_shift_ms", type=float, default=100.0)
    parser.add_argument("--preprocess", type=str, default="mfcc")
    parser.add_argument("--summaries_dir", type=str, default="logs/")
    # --- Redundant TF1-style flags to accept original calls ---
    parser.add_argument("--train_dir", type=str, default="train/")
    parser.add_argument("--verbosity", type=str, default="WARN")
    parser.add_argument("--eval_step_interval", type=int, default=1000)
    parser.add_argument("--save_step_interval", type=int, default=1000)
    FLAGS = parser.parse_args()

    # Prepare words list
    wanted_words = FLAGS.wanted_words.split(",")

    # Prepare model settings
    model_settings = prepare_model_settings(
        len(input_data.prepare_words_list(wanted_words)),
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.dct_coefficient_count,
        FLAGS.preprocess,
    )

    # Initialize FastAudioProcessor
    audio_processor = FastAudioProcessor(
        FLAGS.data_url,
        FLAGS.data_dir,
        FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        wanted_words,
        FLAGS.validation_percentage,
        FLAGS.testing_percentage,
        model_settings,
        None,
    )

    # Time shift in samples
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

    # Create datasets
    train_ds = audio_processor.dataset(
        "training", FLAGS.batch_size,
        FLAGS.background_frequency,
        FLAGS.background_volume,
        time_shift_samples
    )

    val_ds = audio_processor.dataset("validation", FLAGS.batch_size, 0, 0, 0)
    test_ds = audio_processor.dataset("testing", FLAGS.batch_size, 0, 0, 0)

    # Build model
    fingerprint_size = model_settings["fingerprint_size"]
    label_count = model_settings["label_count"]
    model = models.create_model(fingerprint_size, label_count, FLAGS.model_architecture, is_training=True)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(",")))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(",")))
    optimizer = tf.keras.optimizers.Adam(learning_rates_list[0])

    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    step, stage, stage_steps = 0, 0, training_steps_list[0]

    print("Starting training...")
    for features, labels in train_ds.repeat():
        # Update learning rate for staged training
        if step >= stage_steps:
            stage += 1
            if stage < len(training_steps_list):
                stage_steps += training_steps_list[stage]
                optimizer.learning_rate.assign(learning_rates_list[stage])

        # Forward + backward pass
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = loss_fn(labels, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc.update_state(labels, logits)

        if step % 100 == 0:
            print(f"Step {step} | Loss {loss:.4f} | Train Acc {train_acc.result().numpy()*100:.2f}%")
            train_acc.reset_state()

        if step % 1000 == 0:
            for v_features, v_labels in val_ds:
                v_logits = model(v_features, training=False)
                val_acc.update_state(v_labels, v_logits)
            print(f"Validation Accuracy {val_acc.result().numpy()*100:.2f}%")
            val_acc.reset_state()

        step += 1
        if step >= sum(training_steps_list):
            break

    print("\nTraining finished\n")

    # Test set evaluation
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for t_features, t_labels in test_ds:
        logits = model(t_features, training=False)
        test_acc.update_state(t_labels, logits)
    print(f"Final Test Accuracy {test_acc.result().numpy()*100:.2f}%")

if __name__ == "__main__":
    main()