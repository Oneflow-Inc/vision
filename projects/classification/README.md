## Flowvision Image Classification Project
This folder contains the classification project based on `flowvision`

## Usage
<details>
<summary> <b> Installation </b> </summary>

#### Clone flowvision
```bash
git clone https://github.com/Oneflow-Inc/vision.git
cd vision/projects/classification
```

#### Create a conda virtual environment and activate it
```bash
conda create -n oneflow python=3.7 -y
conda activate oneflow
```

#### Install the latest version of OneFlow
```bash
python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/cu102
```

#### Install other requirements
```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

</details>

<details>
<summary> <b> Data preparation </b> </summary>

#### ImageNet
For ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```

### CIFAR100
For CIFAR100, you only need to specify the dataset downloaded path in [config.py](config.py), and set  `DATA.DATASET = 'cifar100'`.

</details>

<details>
<summary> <b> Training </b> </summary>

#### Specify the training model

#### ddp training with simple bash file
```bash
bash ddp_training.sh
```
