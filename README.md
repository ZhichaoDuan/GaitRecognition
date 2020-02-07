# Gait-Recognition-Research

The architecture is mainly inspired by this [repo](https://github.com/L1aoXingyu/Deep-Learning-Project-Template) and I benefit a lot from this way of organizing code.

## Data 
For now, my research mainly focus on `CASIA-B` dataset, which is available [here](http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip)


## Test Environment
This repo works fine on a four NVIDIA 2080Ti machine. All the packages and their versions have been exported to [this file](environment.yml) using

```
conda env export > environment.yml
```

And you can recreate a exact the same environment by 

```
conda env create -f environment.yml
```

## Data Processing
In order to make this model work, we need to align the original data. After downloading CASIA-B dataset, you can use the following command to extract files
```
ls $FOLDER_NAME$ | xargs -n1 -I {} tar zxvf $FOLDER_NAME$/{} -C $TARGET_FOLDER$
```
I used the preprocessing script from this [repo](https://github.com/AbnerHqC/GaitSet). After extracted all the files, you can use the following command to align all the images.
```
python scripts/pretreatment.py --input_path '' --output_path '' --worker_num N
```

## Start Training or Testing
All the config files should be put under folder `configs`. A very simple training example is 
```
python tools/train.py --config_file configs/test.yml MODEL.DEVICE_ID "('3')" LOGGER.NAME TRAIN
```
If you want to test the model generated according to the previous config file, you can type in 
```
python tools/test.py --config_file configs/test.yml MODEL.DEVICE_ID "('3')" LOGGER.NAME TEST
```