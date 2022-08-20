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
pytnon cam.py --gpu 0 --type gradcam
``` 