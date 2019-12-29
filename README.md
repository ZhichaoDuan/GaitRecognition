# Gait-Recognition-Research

The architecture is mainly inspired by this [repo](https://github.com/L1aoXingyu/Deep-Learning-Project-Template) and I benefit a lot from this way of organizing code.

## Data 
For now, my research mainly focus on `CASIA-B` dataset, which is available [here](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)


## Test Environment
This repo works fine on a four NVIDIA 2080Ti machine. All the packages and their versions have been exported to [this file](environment.yml) using

```
conda env export > ${ENV_FILE_NAME}.yml
```

And you can recreate a exact the same environment by 

```
conda env create -f ${ENV_FILE_NAME}.yml
```

## Results
Record all the best results I got so far.

| Configuration | NM | BG | CL | Extra |
| ------------- | --- | --- | --- | --- | 
| <font color=blue>Reported</font> | <font color=blue>95.0</font> | <font color=blue>87.2</font> | <font color=blue>70.4</font> | |
| <font color=red>Baseline with all the same config as the original paper</font> | <font color=red>95.509</font> | <font color=red>89.087</font> | <font color=red>71.318</font> | |
|<font color=black>Baseline with ReLU activation</font>|<font color=black>95.373</font>|<font color=black>88.112</font>|<font color=black>70.045</font>|<font color=black></font>|
|<font color=red> Baseline with ftr size changed to 512 </font>|<font color=red>95.036</font>|<font color=red>88.166</font>|<font color=red>71.245</font>| | 
|<font color=black>Baseline with warmup and weight decay</font>|<font color=black>94.918</font>|<font color=black>86.429</font>|<font color=black>68.727</font>|<font color=black></font>|
|<font color=green> Baseline + BNNeck + neck ftr + cosine dist </font>|<font color=green>96.164</font>|<font color=green>90.306</font>|<font color=green>66.564</font>|<font color=green>ce loss divided by 10, warm up and weight decay added, lr scheduler added, initialization remains unchanged, shift of BN is on, init of fc2 and conv2d is xavier_uniform</font>|
|<font color=red>Baseline + BNNeck + neck ftr + cosine dist</font>|<font color=red>95.127</font>|<font color=red>90.498</font>|<font color=red>71.827</font>|<font color=red>ce loss divided by 10, no warm up and weight decay, no lr scheduler, used pure Adam, initialization changed, shift of BN is disabled, init of fc2 is normal with std 0.001, init of conv2d is kaiming_normal with a=0 and mode='fan_in', all bias of init module is 0.0, init weight of BN is 1.0</font>|
|<font color=black></font>|<font color=black></font>|<font color=black></font>|<font color=black></font>|<font color=black></font>|

## Issues
* If you have no idea how to design your own `collate_fn`, maybe you can check out this [blog](https://www.jianshu.com/p/bb90bff9f6e5)
