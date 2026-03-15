# Casting Defect Detection

This project uses an industrial casting product image dataset to train a deep learning model.
It compares the test results of a model trained on the raw dataset against one trained on the dataset after a custom preprocessing pipeline with OpenCV.

## Prerequisites

### 1. Install `uv`

This project uses `uv` as the dependency manager.
You can install it following the instructions in the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

The most common way to install `uv` is via curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Download the Dataset

Download the [casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) Dataset from Kaggle.
Extract the downloaded archive, navigate into `archive/casting_512x512/casting_512x512` and place the two directories `def_front` and `ok_front` into `data/raw` in the project root.

The directory structure should look like this:

```
data/
├── raw/
│   ├── def_front/
│   │   ├── cast_def_0_0.jpeg
│   │   └── ...
│   └── ok_front/
│       ├── cast_ok_0_35.jpeg
│       └── ...
```
