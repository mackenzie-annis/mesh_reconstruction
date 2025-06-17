# Datasets to Train Ndiffrec
This repo stores the datasets generated using the dataset generation scripts that are needed to construct a 3D model and mesh using Ndifferc

## Folder Structure 

To add your own dataset ensure the file structure is as follows:  

your_dataset/\
├── images/\
│   ├── 00000.jpg\
│   ├── 00001.jpg\
│   └── ...\
├── masks/\
│   ├── 00000.jpg.png\
│   ├── 00001.jpg.png\
│   └── ...\
└── poses.npy\
└── config_your_dataset.json\





