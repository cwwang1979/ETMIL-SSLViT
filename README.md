# ETMIL-OMF

## Dataset
We utilized endometrial cancer and colorectal cancer anonymized H\&E-stained WSIs collected from The Cancer Genome Atlas (TCGA) cohort. TCGA is a comprehensive repository of tissue specimens from 30 tissue source sites available in the public repositories at the National Institutes of Health, USA ([TCGA Link](https://portal.gdc.cancer.gov/))


## Setup

#### Requirerements
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on a single Nvidia GeForce GTX 1080)
- Python (3.7.11), h5py (2.10.0), opencv-python (4.2.0.34), PyTorch (1.10.1), torchvision (0.11.2), pytorch-lightning (1.2.3).

#### Download
Source code file, configuration file, and models are download from the [zip](https://drive.google.com/file/d/1nzHdmnrSw_1-m4KuF26vVRJRcTzgCRiR/view?usp=drive_link) file.  (For reviewers, the password of the file is in the implementation section of the associated manuscript.)

## Steps
#### 1.Installation

Please refer to the following instructions.
```
# create and activate the conda environment
conda create -n tmil python=3.7 -y
conda activate transmil

# install pytorch
## pip install
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
## conda install
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# install related package
pip install -r requirements.txt
```

#### 1. Tissue Segmentation and Patching

Place the Whole slide image in ./DATA
```
./DATA/XXXX
├── slide_1.svs
├── slide_2.svs
│        ⋮
└── slide_n.svs
  
```

Then in a terminal run:
```
python create_patches.py --source DATA/XXXX --save_dir DATA_PATCHES/XXXX --patch_size 256 --preset tcga.csv --seg --patch --stitch

```

After running in a terminal, the result will be produced in folder named 'DATA_PATCHES/XXXX', which includes the masks and the sticthes in .jpg and the coordinates of the patches will stored into HD5F files (.h5) like the following structure.
```
DATA_PATCHES/XXXX/
├── masks/
│   ├── slide_1.jpg
│   ├── slide_2.jpg
│   │       ⋮
│   └── slide_n.jpg
│
├── patches/
│   ├── slide_1.h5
│   ├── slide_2.h5
│   │       ⋮
│   └── slide_n.h5
│
├── stitches/
│   ├── slide_1.jpg
│   ├── slide_2.jpg
│   │       ⋮
│   └── slide_n.jpg
│
└── process_list_autogen.csv
```


#### 2. Feature Extraction

In the terminal run:
```
CUDA_VISIBLE_DEVICES=0,1 python extract_features.py --data_h5_dir DATA_PATCHES/XXXX/ --data_slide_dir DATA/XXXX --csv_path DATA_PATCHES/XXXX/process_list_autogen.csv --feat_dir DATA_FEATURES/XXXX/ --batch_size 512 --slide_ext .svs

```

example features results:
```
FEATURES_DIRECTORY_RESNET152/
├── h5_files/
│   ├── slide_1.h5
│   ├── slide_2.h5
│   │       ⋮
│   └── slide_n.h5
│
└── pt_files/
    ├── slide_1.pt
    ├── slide_2.pt
    │       ⋮
    └── slide_n.pt
```

#### 3. Training and Testing List
Prepare the training, validation  and the testing list containing the labels of the files and put it into ./dataset_csv folder. (We provides the csv sample training and testing list in named "fold0.csv")

example of the csv files:
|      | train          | train_label     | val        | val_label | val        | val_label |  
| :--- | :---           |  :---           | :---:      |:---:      | :---:      |:---:      | 
|  0   | slide_1        | 1               | slide_1    |   0       | slide_1    |   0       | 
|  1   | slide_2        | 0               | slide_2    |   1       | slide_2    |   0       |
|  ... | ...            | ...             | ...        | ...       | ...        | ...       |
|  n   | slide_n        | 1               | slide_n    |   1       | slide_n    |   1       |



#### 4. Inference 

Run this code in the terminal to ensemble the results of the top K models:
```
python ensemble_inf.py --stage='test' --config='Config/TMIL.yaml'  --gpus=0 --top_fold=K
```

## Training
#### Preparing Training Splits

To create a N fold for training and validation set from the training list. The default proportion for the training:validation splits used in this study is 9:1. 
```
dataset_csv/
├── fold0.csv
├── fold1.csv
│       ⋮
└── foldN.csv
```

#### Training

Run this code in the terminal to training:
```
python ensemble_inf.py --stage='test' --config='Config/TMIL.yaml'  --gpus=0 --top_fold=Z
```

## License
This Python source code is released under a creative commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology

