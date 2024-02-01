## Data Preparation

Crop the training and validating data before the training process.

### Real Image Denoising
1. Please download the corresponding training datasets and put them in the folder `datasets\SIDD`.
2. Run the instruction here.
```bash
python generate_patches_sidd.py
```

### Single image motion deblurring
1. Please download the corresponding training datasets and put them in the folder `datasets\GoPro`.
2. Run the instruction here.
```bash
python generate_patches_gopro.py
```

### Gaussion Image Denoising
We do not crop the training data in our implementation.

