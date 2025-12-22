# model_definition.py

import numpy as np
import onnxruntime as ort
from pathlib import Path

# -----------------------------
# Load ONNX model ONCE
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = BASE_DIR / "artifacts" / "cnn_text_classifier.onnx"

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# -----------------------------
# Prediction function
# -----------------------------

def predict(vector: np.ndarray) -> dict:
    """
    vector shape: (embedding_dim,)
    """

    if vector.ndim != 1:
        raise ValueError("Input vector must be 1D")

    # Replace NaNs if any (safety)
    vector = np.nan_to_num(vector)

    # Reshape -> (batch_size, channels, embedding_dim)
    vector = vector.reshape(1, 1, -1).astype(np.float32)

    # Run ONNX inference
    output = session.run(
        [output_name],
        {input_name: vector}
    )[0]

    probability = float(output[0][0])
    label = int(probability >= 0.5)

    return {
        "label": label,
        "probability": probability
    }
