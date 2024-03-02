# TUCKET

A tensor time series data structure for efficient and accurate factor analysis over time ranges

## Datasets

Please put the datasets in the folder `inputs/`.

**Note:** Due to the large sizes of datasets, please email `rq5 AT illinois DOT edu` to request for the datasets.

## Dependencies

The code was tested under the following dependencies:

- Python 3.10.13
- NumPy 1.26.3
- PyTorch 2.1.2
- CUDA 12.2

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
