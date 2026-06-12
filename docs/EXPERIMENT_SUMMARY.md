# Experiment Summary

This project studies a multimodal image retrieval agent that routes user
requests through an intent-recognition module and then calls image retrieval
modules when needed.

## System Modules

- LLM intent agent: classifies text-only user questions into text-to-image
  retrieval, image-to-image retrieval, or no retrieval.
- VLM intent agent: handles image + text questions and decides whether the image
  should be used as retrieval context.
- CLIP retrieval: supports text-to-image, image-to-image, and text-image fused
  retrieval.
- Fine-grained retrieval: uses Qwen2.5-VL to produce image labels, encodes the
  labels with Taiyi-CLIP, and searches the label-feature space.
- LoRA fine-tuning: fine-tunes Qwen2.5-VL label generation for the selected
  categories.

## Key Results Recorded In The Experiment

- Taiyi-CLIP performed best among the tested Chinese/English CLIP baselines for
  the text-to-image setup.
- Qwen2.5-7B with few-shot prompting achieved strong intent-recognition results,
  with average F1 around 0.967 in the recorded experiment.
- Qwen2.5-VL-7B achieved comparable multimodal intent-recognition performance,
  with average F1 around 0.966 in the recorded experiment.
- Qwen2.5-VL LoRA fine-tuning improved fine-grained label retrieval in the
  regular subset, with recorded average F1 around 0.884 for text-to-image
  retrieval after fine-tuning.

This English summary captures the main experiment design and results for the
public portfolio version of the project.
