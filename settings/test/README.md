## Configurations

For different tasks, you should choose different .json file to initialize the corresponding settings. And the main settings of all json files are as follows:
+ basic settings:

```c++
 "name": "sr_bd/0_15"                          //name to save results
  , "model":"CFSNet"
  , "task_type": "sr"                          //sr, deblock, denoise
  , "scale": 3                                 //upscale for SR (x4 for SR and x3 for SR+Deblur in our paper)
  , "input_alpha":0.15                         //control the variable alpha_in to get different image restoration results
  , "noise_level": 50                          //the noise level of input images
  , "gpu_ids": [3]
  , "is_train": false
  , "use_gan": false
  , "use_tensorboard_logger": false
```
  
+ path:
  
```c++
   , "path": {
    "root": "/media/data1/CFSNet/test/"        // path to save results
    , "saved_model": "../models/latest_G.pth"  // path of the trained model
    , "log": null                              //initialized in options.py
    , "results_root": null                     //initialized in options.py
  }
```
  
+ datasets:

```c++
  , "datasets": {
    "test_1": {
      "name": "PIRMTest" 
      , "mode": "sr"                           // the type of datasets
      , "dataroot_GT": "../test/PIRMTest/Original"
      , "dataroot_LR": "../test/PIRMTest/4x_downsampled"
    }
  "test_1": {
    "name": "Set12"
    , "mode": "denoise"                        // the type of datasets
    , "dataroot_GT": "../data/Set12"
    , "color": "gray"                          //gray, RGB
    }
  }
```

+ network:

```c++
 , "network_G": {
    "in_channel": 3                           //the number of the input channel
    , "out_channel": 3                        // the number of the output channel
    , "num_channels": 64                      // features channel number
    , "n_main_b": 10                          //the number of the main blocks in the main branch
    , "n_tuning_b": 10                        //the number of the tuning blocks in the tuning branch
  }
```
