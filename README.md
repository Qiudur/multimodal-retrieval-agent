# Multimodal Retrieval Agent

An experimental multimodal image retrieval agent for image databases. The
project includes LLM/VLM intent recognition, Taiyi-CLIP text-image and
image-image retrieval, Qwen2.5-VL fine-grained image-label understanding, and
Qwen2.5-VL LoRA fine-tuning experiments.

## Project Highlights

- Text intent routing: determines whether a text-only user query needs image
  retrieval, and routes it to text-to-image retrieval, image-to-image retrieval,
  or no retrieval.
- Multimodal intent routing: handles image + text inputs and decides whether the
  image should be used as retrieval context.
- CLIP retrieval: uses Taiyi-CLIP / CLIP to encode text and images, supporting
  Top-K retrieval and threshold-based evaluation.
- Fine-grained retrieval: uses Qwen2.5-VL to generate image-element labels, then
  encodes those labels with Chinese CLIP features for retrieval.
- LoRA experiments: fine-tunes Qwen2.5-VL for fine-grained image-label
  generation.

## Repository Layout

```text
configs/                    # path and model configuration examples
data/metadata/              # experiment metadata and generated labels
docs/                       # English experiment summary
notebooks/                  # cleaned English experiment notebooks
scripts/                    # command-line helper scripts
src/multimodal_retrieval_agent/
  intent/                   # intent schema and lightweight baseline
  retrieval/                # optional Taiyi-CLIP retriever
  evaluation/               # metrics and intent evaluation
tests/                      # minimal tests
```

## Quick Start

Create and activate a Python environment, then install the project in editable
mode.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the lightweight intent agent.

```bash
mmra intent "Please show me three chair photos"
mmra intent "Find images similar to this skateboard photo" --has-image
```

Evaluate the rule-based intent baseline on the generated intent dataset.

```bash
mmra eval-intent --dataset data/metadata/gpt_intent.json
```

Check whether local image files are available.

```bash
mmra check-data --metadata data/metadata/image_dataset.json
```

The included metadata points to `data/images/<category>/<file_name>`. Original
ImageNet/COCO images are not included because of dataset licensing and size.
See [data/README.md](data/README.md) for the expected layout.

## Heavy Model Setup

The basic CLI works without GPU. To run CLIP, Qwen, Qwen-VL, or LoRA notebooks,
install the optional model dependencies.

```bash
pip install -e ".[models]"
```

For Qwen-VL fine-tuning, this project keeps the modified training script in
`scripts/train_qwen_lora.py`, but it still depends on the official Qwen-VL
fine-tuning package structure. Recommended workflow:

1. Clone the official Qwen-VL/Qwen3-VL repository.
2. Install its fine-tuning dependencies.
3. Use this repository's `scripts/train_qwen_lora.py` as the modified training
   entry that supports selecting LoRA target modules.
4. Register your dataset path using the Qwen-VL data config.

## Notebooks

- `01_taiyi_clip_threshold.ipynb`: Taiyi-CLIP text/image/fused retrieval
  threshold and Top-K evaluation.
- `02_qwen_labels_f1.ipynb`: Qwen2.5-VL label extraction and fine-grained
  retrieval evaluation.
- `03_intent_recognition.ipynb`: GPT-generated intent dataset and Qwen/Qwen-VL
  intent experiments.
- `04_qwen_vl_lora_train.ipynb`: Qwen2.5-VL LoRA fine-tuning experiments.

All notebooks were rewritten as English public-facing experiment notebooks
before release. API keys are represented as environment variables, and personal
Google Drive paths were replaced by relative or placeholder paths.

## Environment Variables

Only needed when regenerating GPT-based intent data.

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Only needed when using local model copies.

```bash
export QWEN_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"
export QWEN_VL_MODEL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
export QWEN_LORA_ADAPTER_PATH="/path/to/lora_adapter"
```

## Reproducibility Notes

This is an experimental research project. The repository includes cleaned code,
metadata, generated labels, result documentation, and a lightweight runnable
baseline. Full reproduction of the reported CLIP/Qwen/Qwen-VL results requires:

- downloading ImageNet/COCO images according to the metadata;
- downloading the referenced Hugging Face/ModelScope models;
- running the notebooks on a GPU environment such as Colab, A100, L4, or similar;
- adapting Qwen-VL fine-tuning paths to your local environment.

## Portfolio Notes

The project was originally developed from Chinese research notes, but this
release directory has been converted to an English-facing portfolio version:
public code, metadata, schemas, prompts, notebooks, and documentation use
English names and English task definitions.

## License

Code in this repository is released under the MIT License. Dataset images and
pretrained model weights follow their original licenses.
