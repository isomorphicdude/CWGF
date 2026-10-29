# Downloading Datasets

The datasets are stored on Hugging Face in `lmdb` format. Download them with the `hf` CLI (installed with the environment) into the folders below, which are the roots used by `configs/dataset/`.

## FFHQ
To download the FFHQ validation dataset (512 x 512, 3.9 GB; `dataset=ffhq512_val`, used by `scripts/eval.sh`), run the following command:
```bash
hf download isometricneko/FFHQ512-val-lmdb --repo-type dataset --local-dir data/ffhq512_val
```
For the 1024 x 1024 version (13.7 GB; `dataset=ffhq1024_val`):
```bash
hf download isometricneko/FFHQ1024-val-lmdb --repo-type dataset --local-dir data/ffhq1024_val
```

## AFHQ
To download the AFHQ cat/dog validation dataset (512 x 512, 0.4 GB; `dataset=afhq512_val`), run the following command:
```bash
hf download isometricneko/AFHQ512-val-lmdb --repo-type dataset --local-dir data/afhq512-catdog-val
```

## ImageNet
To download the ImageNet validation dataset (uncropped, 6.8 GB; `dataset=imagenet512_val`) in `lmdb` format, run the following command:
```bash
hf download isometricneko/ImageNet-val-lmdb --repo-type dataset --local-dir data/imagenet512_val
```
Experiments use the 1000-image subset listed in `data/ctest1k.csv`.

## MS-COCO 2017
To download the MS-COCO 2017 validation dataset (uncropped, 0.8 GB; `dataset=mscoco2017_val`) in `lmdb` format, run the following command:
```bash
hf download isometricneko/MS-COCO-2017-val --repo-type dataset --local-dir data/ms-coco-val-2017
```
