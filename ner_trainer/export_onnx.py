"""Export fine-tuned PyTorch NER model to optimized INT8 ONNX format for Android."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForTokenClassification, AutoTokenizer

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

MODEL_DIR = Path(__file__).resolve().parent / "model"
LABEL_LIST = ["O", "B-TERM", "I-TERM"]


def export_to_onnx(model_dir: Path, output_onnx: Path):
    """Export PyTorch HuggingFace model to standard ONNX."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    dummy_text = "tell me what does kick the bucket mean"
    inputs = tokenizer(dummy_text, return_tensors="pt")

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    print(f"Exporting PyTorch model to ONNX: {output_onnx}...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        f=str(output_onnx),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        dynamo=False,
    )
    print("ONNX export complete.")


def quantize_model(input_onnx: Path, quantized_onnx: Path):
    """Quantize 32-bit float ONNX model to 8-bit integer (INT8) for mobile."""
    print(f"Quantizing {input_onnx} to INT8: {quantized_onnx}...")
    quantize_dynamic(
        model_input=str(input_onnx),
        model_output=str(quantized_onnx),
        weight_type=QuantType.QInt8,
    )
    fp32_size = input_onnx.stat().st_size / (1024 * 1024)
    int8_size = quantized_onnx.stat().st_size / (1024 * 1024)
    print(f"Quantization complete! FP32 size: {fp32_size:.1f} MB -> INT8 size: {int8_size:.1f} MB")


def test_onnx_inference(onnx_path: Path, model_dir: Path):
    """Run test predictions on the quantized model."""
    print(f"\n--- Testing ONNX Model: {onnx_path.name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(str(onnx_path))

    test_queries = [
        "tell me what is black mean",
        "what does kick the bucket mean",
        "can you define serendipity for me please",
        "what is the weather today in new york",
        "meaning of ran",
        "i was reading and saw the word ubiquitous what is that",
    ]

    for query in test_queries:
        t0 = time.perf_counter()
        inputs = tokenizer(query, return_tensors="np")
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        outputs = session.run(["logits"], ort_inputs)
        logits = outputs[0][0]
        pred_ids = np.argmax(logits, axis=-1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract labeled tokens
        extracted_tokens = []
        for token, pred_id in zip(tokens, pred_ids):
            tag = LABEL_LIST[pred_id]
            if tag in ("B-TERM", "I-TERM"):
                extracted_tokens.append(token)

        extracted_text = tokenizer.convert_tokens_to_string(extracted_tokens).strip()
        print(f"Query:   '{query}'")
        print(f"Result:  '{extracted_text or None}'  ({latency_ms:.1f} ms)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Export NER model to ONNX for Android")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    fp32_onnx = args.model_dir / "term_extractor.onnx"
    int8_onnx = args.model_dir / "term_extractor_int8.onnx"

    export_to_onnx(args.model_dir, fp32_onnx)
    quantize_model(fp32_onnx, int8_onnx)
    test_onnx_inference(int8_onnx, args.model_dir)


if __name__ == "__main__":
    main()
