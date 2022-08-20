# cam-variants

### Install Conda Dependencies
```
conda env create -f environment.yml
conda activate cam
```
### Run CAM
```
# To view optional arguments
python cam.py --help

# Run example
python cam.py --gpu 0 --type gradcam
``` 