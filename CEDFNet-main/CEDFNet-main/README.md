# Paper
Our method is a two-stage underwater image enhancement method. You can follow the following instructions to perform training, prediction and evaluation metric calculations:
## First Stage
Please note the following points:
1. Make sure your input image is 128*128 in size, which you can obtain by modifying the generate-patches method.
2. Please modify the parameters in options.py according to your actual situation to configure the training environment.
3. You can train through the train.py file.
4. Predictions can be made through GetClearPic.py.

## Second Stage
Please note the following points:
1. Please modify the parameters in config/underwater.json according to your actual situation to configure the training environment.
2. You can train through the train.py file.
3. You can predict through infer.py.

## Evaluation Metric Calculations
1. Change the input_images_path and image2smiles2image_save_path parameters in the PSNR-SSIM-LPIPS.py file, and then run to get PSNR, SSIM and LPIPS
2. You can calculate fid using  'python -m metrics.fid --path yourpath1 yourpath2'

You can get the pre-trained results from the following link  
https://pan.baidu.com/s/1PDeFSJAV-I1xF6oYtA3ygQ?pwd=h4zj

