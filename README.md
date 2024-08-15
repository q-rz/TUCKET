# TUCKET (VLDB 2025)

A tensor time series data structure for efficient and accurate factor analysis over time ranges

![Illustration of TUCKET](https://raw.githubusercontent.com/q-rz/TUCKET/main/fig-rqa.svg)

## Dependencies

The code was tested under the following dependencies:

- Python 3.10.13
- NumPy 1.26.3
- PyTorch 2.1.2
- CUDA 12.2

## Datasets

Please download datasets and corresponding queries at [our HuggingFace repo](https://huggingface.co/datasets/q-rz/VLDB25-TUCKET) and put them into the folder `inputs/`.

## Usage

To reproduce our results, please run:

```bash
chmod +x scripts/*.sh
./scripts/test_{method}_{dataset}.sh {device}
```

where:
- `{method}` can be `tuckerals` / `dtucker` / `zoomtucker` / `tucket` / `tucket_zoomtuckerstitch`;
- `{dataset}` can be `airquality` / `traffic` / `usstock` / `krstock`;
- `{device}` is the device for PyTorch (e.g., `cuda:0` or `cpu`).
