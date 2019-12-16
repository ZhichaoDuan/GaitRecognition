# Gait-Recognition-Research

## Data 
For now, my research mainly focus on `CASIA-B` dataset, which is available [here](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)


## Test Environment
This repo works fine on a four NVIDIA 2080Ti machine. All the packages and their versions have been exported to [this file](environment.yml) using

```
conda env export > ${ENV_FILE_NAME}.yaml
```

And you can recreate a exact the same environment by 

```
conda env create -f ${ENV_FILE_NAME}.yaml
```

## Issues
* If you have no idea how to design your own `collate_fn`, maybe you can check out this [blog](https://www.jianshu.com/p/bb90bff9f6e5)
