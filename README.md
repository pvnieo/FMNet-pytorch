# FMNet-pytorch
A pytorch implementation of: "Deep Functional Maps: Structured Prediction for Dense Shape Correspondence" [[link](http://openaccess.thecvf.com/content_ICCV_2017/papers/Litany_Deep_Functional_Maps_ICCV_2017_paper.pdf)]

## Installation
This implementation runs on python >= 3.7, use pip to install dependencies:
```
pip3 install -r requirements.txt
```

## Download data & preprocessing
Download the desired dataset and put it in the `data` folder. Multiple datasets are available [here](https://github.com/pvnieo/datasets-zoo).

<ins>An example with the faust-remeshed dataset is provided</ins>.

Build shot calculator:
```
cd fmnet/utils/shot
cmake .
make
```
If you got any errors in compiling shot, please see [here](https://github.com/pvnieo/3d-utils/tree/master/shot).

Use `fmnet/preprocess.py` to calculate the Laplace decomposition, geodesic distance using the Dijkstra algorithm and the shot descriptors of input shapes, data are saved in .mat format:
```
usage: preprocess.py [-h] [-d DATAROOT] [-sd SAVE_DIR] [-ne NUM_EIGEN] [-nj NJOBS] [--nn NN]

Preprocess data for FMNet training. Compute Laplacian eigen decomposition, shot features, and geodesic distance for each shape.

optional arguments:
  -h, --help            show this help message and exit
  -d DATAROOT, --dataroot DATAROOT
                        root directory of the dataset
  -sd SAVE_DIR, --save-dir SAVE_DIR
                        root directory to save the processed dataset
  -ne NUM_EIGEN, --num-eigen NUM_EIGEN
                        number of eigenvectors kept.
  -nj NJOBS, --njobs NJOBS
                        Number of parallel processes to use.
  --nn NN               Number of Neighbor to consider when computing geodesic matrix.
```
**NB**: if the shapes have many vertices, the computation of geodesic distance will consume a lot of memory and take a lot of time.

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