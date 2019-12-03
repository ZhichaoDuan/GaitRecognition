# Gait-Recognition-Research

## Test Environment
This repo works fine on a four NVIDIA 2080Ti machine. All the packages and their versions have been exported to [this file](environment.yaml) using

```
conda ${ENV_NAME} export > ${ENV_FILE_NAME}.yaml
```

And you can recreate a exact the same environment by 

```
conda ${ENV_NAME} create -f ${ENV_FILE_NAME}.yaml
```
