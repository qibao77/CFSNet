For different tasks, you should choose different .json file to initialize the corresponding settings. Some specific settings for different tasks are as follows:

### Super-resolution:
+ Step 1 (train the main branch):
```c++
"name": "sr/step1"                         // name to save training results
, "scale": 4                               // upscale for SR 
 , "input_alpha":0                         // set the control variable α_in as 0
, "use_gan": false                         // we use mae loss in Step 1
, "saved_model": null                      // path of the trained model
```

+ Step 2 (train the tuning branch):
```c++
"name": "sr/step2"                         // name to save training results
, "scale": 4                               // upscale for SR 
 , "input_alpha":1                         // set the control variable α_in as 1
, "use_gan": true                          // we use mae loss , perceptual loss and gan loss in Step 2
, "saved_model": "../models/latest_G.pth"  // path of the model trained in Step 1.
```

### Denoising:
+ Step 1 (train the main branch):
```c++
"name": "denoise_gray/sigma25_step1"       // name to save training results
 , "input_alpha":0                         // set the control variable α_in as 0
, "noise_level": 25                        // the corresponding noise level of the input images
, "use_gan": false                         // we use mse loss in Step 1
, "saved_model": null                      // path of the trained model
```

+ Step 2 (train the tuning branch):
```c++
"name": "denoise_gray/sigma50_step2"        // name to save training results
 , "input_alpha":1                          // set the control variable α_in as 1
, "noise_level": 50                         // the corresponding noise level of the input images
, "use_gan": false                          // We use the same loss as used in Step 1
, "saved_model": "../models/latest_G.pth"   // path of the model trained in Step 1.
```

### Deblocking:
+ Step 1 (train the main branch):
```c++
"name": "deblock/step1"                    // name to save training results
 , "input_alpha":0                         // set the control variable α_in as 0
, "use_gan": false                         // we use mae loss in Step 1
, "saved_model": null                      // path of the trained model
, "dataroot_LR": "../data/BSD500/train400_Y_jpeg/jpeg_10" // the corresponding input images with  quality factor 10
```

+ Step 2 (train the tuning branch):
```c++
"name": "deblock/step2"                    // name to save training results
 , "input_alpha":1                         // set the control variable α_in as 1
, "use_gan": false                         // We use the same loss as used in Step 1
, "saved_model": "../models/latest_G.pth"  // path of the model trained in Step 1.
, "dataroot_LR": "../data/BSD500/train400_Y_jpeg/jpeg_40" // the corresponding input images with  quality factor 40
```
