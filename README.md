## MRI super-resolution

### Table of content
- [MRI super-resolution](#mri-super-resolution)
  - [Table of content](#table-of-content)
- [1. Installation](#1-installation)
- [2. Dataset](#2-dataset)
- [3. Train model](#3-train-model)
- [4. Monitoring and Visualization](#4-monitoring-and-visualization)
- [5. Prediction](#5-prediction)

## 1. Installation
- Create new conda environment `supermri`
  ```
  conda create -n supermri python=3.8
  ```
- Activate `supermri` environment
  ```
  conda activate supermri
  ```
- Run installation script
  ```
  sh setup.sh
  ```
- Restart conda environment after installation.

## 2. Dataset
- See [data/RAEDME.md](data/README.md) for the structure of the dataset and how to use the pre-processing script.

## 3. Train model
- `train.py` is the main file to train different models and datasets, it also supports `tensorboard` monitoring, checkpoint saving and mixed precision training. The following is an example on how to train a `UNet` on `MDS`.
- Activate `supermri` conda environment
  ```
  conda activate supermri
  ```
- The following command train a UNet on `MDS` for 100 epochs, and store model checkpoint and summary to `runs/001_unet_mds`
  ```
  python train.py --input_dir data/dataset/MDS --output_dir runs/001_unet_mds --model unet --num_filters 64 --patch_dim 32 --n_patches 1000 --merge_scan_type --batch_size 32 --epochs 100 --mixed_precision
  ```
- To see all the input arguments, use the `--help` flag.
  ```
  python train.py --help
  ```

## 4. Monitoring and Visualization
- To monitor training performance and up-sampled results stored in `--output_dir`, we can use the following command
  ```
  tensorboard --logdir runs/001_unet_mds
  ```
- By default, TensorBoard starts a local server at port `6006`, you can check the TensorBoard summary by visiting `localhost:6006`.

## 5. Prediction
- After training the model with checkpoint saved at `--output_dir`, you can use `predict.py` to up-sample any scans in `.mat` format.
- For instance, if scans needed to be up-sampled are stored in `data/test_scans` then we can use the following command
  ```
  python predict.py --input_dir data/test_scans --output_dir runs/001_unet_mds --upsample_dir data/test_scans/upsample
  ```
- The up-sampled scans would be stored under `--upsample_dir` with the same name as the original scan.
- Note that scans have keys: `{FLAIRarray, T1array, T2array}` in the `.mat` file.
