# Lightweight NER Model Trainer for Android

This folder contains a pipeline to generate training data, fine-tune a tiny mobile transformer (`MiniLM` or `MobileBERT`), and export it as an optimized 8-bit quantized ONNX model for Android.

## Workflow

### 1. Install Training Dependencies
```bash
pip install -r ner_trainer/requirements.txt
```

### 2. Generate Dataset (Already Done!)
Generates 10,000 synthetic queries (single words, multi-word idioms, and negative chit-chat queries) sampled from `dictionary.db`:
```bash
python ner_trainer/generate_dataset.py
```
Outputs:
- `ner_trainer/data/train.jsonl` (8,500 samples)
- `ner_trainer/data/val.jsonl` (1,500 samples)

### 3. Fine-Tune Model
Fine-tunes a ~22M parameter transformer for token classification:
```bash
python ner_trainer/train.py --epochs 3 --batch-size 32
```
On GPU or modern CPU, training takes ~3–8 minutes.
Saves model to `ner_trainer/model/`.

### 4. Export & Quantize for Android
Converts the trained model to INT8 ONNX format:
```bash
python ner_trainer/export_onnx.py
```
Outputs:
- `ner_trainer/model/term_extractor_int8.onnx` (~15–20 MB)
- Evaluates test queries and prints sub-10ms inference latencies.

## Using in Android
Drop `term_extractor_int8.onnx` into `app/src/main/assets/` and run using `com.microsoft.onnxruntime:onnxruntime-android`.
