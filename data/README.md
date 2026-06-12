# Data

This repository includes metadata and generated labels, but it does not include
the original ImageNet/COCO images.

## Expected Image Layout

```text
data/images/
  chair/
  cup/
  bicycle/
  skateboard/
  oven/
  ...
```

The metadata files use relative paths such as
`data/images/chair/n03001627_7.JPEG`. Put your downloaded images under the same
layout, or update `configs/paths.example.yaml`.

## Dataset Design

- 2,000 images in total.
- 1,000 regular ImageNet-style images and 1,000 fine-grained COCO images.
- 5 positive categories: chair, cup, bicycle, skateboard, oven.
- 10 negative categories: couch, bowl, motorcycle, snowboard, microwave, cat,
  donut, keyboard, orange, pizza.
- The positive-to-negative ratio is 1:9.
