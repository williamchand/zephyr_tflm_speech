"""
Creates hardlinks from your source files into the tflite-micro submodule.
Run this once after cloning, then use bazel/make commands directly.
"""
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
TFLM = REPO_ROOT / "tflite_micro/tensorflow/lite/micro/examples/micro_speech"

FILES_TO_LINK = [
    (REPO_ROOT / "micro_model_settings.h",
     TFLM / "micro_model_settings.h"),

    (REPO_ROOT / "micro_speech_test.cc",
     TFLM / "micro_speech_test.cc"),

    (REPO_ROOT / "evaluate.py",
     TFLM / "evaluate.py"),

    (REPO_ROOT / "evaluate_test.py",
     TFLM / "evaluate_test.py"),

    (REPO_ROOT / "models/micro_speech_quantized.tflite",
     TFLM / "models/micro_speech_quantized.tflite"),

    (REPO_ROOT / "train/train_micro_speech_model.ipynb",
     TFLM / "train/train_micro_speech_model.ipynb"),
]

def main():
    # check submodule is initialized
    if not (REPO_ROOT / "tflite_micro/.git").exists():
        print("ERROR: tflite_micro submodule not initialized.")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    print("==> Setting up links into tflite_micro...\n")
    for src, dst in FILES_TO_LINK:
        if not src.exists():
            print(f"  ERROR: source not found: {src}")
            sys.exit(1)

        if dst.exists():
            dst.unlink()

        try:
            os.link(src, dst)
            print(f"  linked:  {src.name}")
        except OSError:
            # fallback for cross-drive on Windows
            shutil.copy2(src, dst)
            print(f"  copied:  {src.name} (cross-drive fallback)")

    print("\n==> Done. You can now run:")
    print()
    print("  cd tflite_micro")
    print()
    print("  # Bazel")
    print("  bazel run tensorflow/lite/micro/examples/micro_speech:micro_speech_test")
    print()
    print("  # Make")
    print("  make -f tensorflow/lite/micro/tools/make/Makefile test_micro_speech_test")
    print()
    print("  # Make QEMU Cortex-M0 + CMSIS-NN")
    print("  make -f tensorflow/lite/micro/tools/make/Makefile \\")
    print("    TARGET=cortex_m_qemu TARGET_ARCH=cortex-m0 \\")
    print("    OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=default \\")
    print("    test_micro_speech_test")

if __name__ == "__main__":
    main()