# FMNet-pytorch
A pytorch implementation of: "Deep Functional Maps: Structured Prediction for Dense Shape Correspondence" [[link](http://openaccess.thecvf.com/content_ICCV_2017/papers/Litany_Deep_Functional_Maps_ICCV_2017_paper.pdf)]

## Installation
This implementation runs on python >= 3.7, use pip to install dependencies:
```
pip3 install -r requirements.txt
```

## Download data & preprocessing
Download the FAUST dataset (or other) from original [website](http://faust.is.tue.mpg.de).

Build shot calculator:
```
cd data/shot
cmake .
make
```

Calculate eigenvectors, geodesic maps, shot descriptors of trained models, save in .mat format:
```
usage: preprocess.py [-h] [-d DATAROOT] [-sd SAVE_DIR] [-ne NUM_EIGEN]
                     [--no-shot] [--no-geo] [--no-dec]

Preprocess data for FMNet training. Compute Laplacian eigen decomposition,
shot features, and geodesic distance for each shape.

optional arguments:
  -h, --help            show this help message and exit
  -d DATAROOT, --dataroot DATAROOT
                        root directory of the dataset
  -sd SAVE_DIR, --save-dir SAVE_DIR
                        root directory to save the computed matrices
  -ne NUM_EIGEN, --num-eigen NUM_EIGEN
                        number of eigenvectors kept.
  --no-shot             Do not compute shot features.
  --no-geo              Do not compute geodesic distances.
  --no-dec              Do not compute Laplacian eigen decomposition.
```

## Usage
Use the `train.py` script to train the FMNet network.
```
usage: train.py [-h] [--lr LR] [--b1 B1] [--b2 B2] [-bs BATCH_SIZE] [--n-epochs N_EPOCHS] [--feat-dim FEAT_DIM] [-nv N_VERTICES] [-nb NUM_BLOCKS] [-d DATAROOT] [--save-dir SAVE_DIR]
                [--n-cpu N_CPU] [--no-cuda] [--checkpoint-interval CHECKPOINT_INTERVAL] [--log-interval LOG_INTERVAL]

Lunch the training of FMNet model.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               adam: learning rate
  --b1 B1               adam: decay of first order momentum of gradient
  --b2 B2               adam: decay of first order momentum of gradient
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        size of the batches
  --n-epochs N_EPOCHS   number of epochs of training
  --feat-dim FEAT_DIM   Input feature dimension
  -nv N_VERTICES, --n-vertices N_VERTICES
                        Number of vertices used per shape
  -nb NUM_BLOCKS, --num-blocks NUM_BLOCKS
                        number of resnet blocks
  -d DATAROOT, --dataroot DATAROOT
                        root directory of the dataset
  --save-dir SAVE_DIR   root directory of the dataset
  --n-cpu N_CPU         number of cpu threads to use during batch generation
  --no-cuda             Disable GPU computation
  --checkpoint-interval CHECKPOINT_INTERVAL
                        interval between model checkpoints
  --log-interval LOG_INTERVAL
                        interval between logging train information
```

### Example
```
python3 train.py -d ./data/faust/train_mini -bs 2 -n-epochs 2
```