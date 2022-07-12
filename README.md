# 3D Vessel Tree Generator

This repository can be used to generate random vessels or vessel trees. 
Vessel trees are encoded and saved as an MxNX4 matrix, where M is the number of branches in the tree, 
N is the number of interpolated centerline points per branch, and the last dimension corresponds to the (x,y,z) coordinates and radius r for each centerline point.

The vessel tree generator is highly customizable. Many parameters are randomized by default but can be specified if the user would like more control over the geometries.
Parameters include:
 - vessel tree type: cylinders, random splines, or right coronary tree
 - vessel dimensions including length and maximum radius
 - constant radius, linearly tapered radius, or user-specified radius
 - number, position, and severity of stenoses
 - relative positions and dimensions of side branches
 
An info file is saved for each geometry which contains the parameters used to construct it.

## Dependencies
numpy\
matplotlib\
scikit-image (skimage)\
[NURBS-python](https://nurbs-python.readthedocs.io/en/5.x/install.html) (geomdl)

## Usage

```commandline
python ./tube_generator.py --save_path="/path/to/save" --dataset_name="test" --num_trees=10 --num_branches=3 --vessel_type='RCA' --shear
```

`tube_generator.py` contains optional code to generate random binary projections of the 3D geometries.

### To do:

- implement left coronary tree, other vessels
- implement cosine profile stenoses

## Citation
If you find this work useful, please cite the following:

**A Machine Learning Approach for Coronary 3D Reconstruction from Uncalibrated X-ray Angiography Images. Iyer, K., Nallamothu, B.K., Figueroa, C.A., Nadakuditi, R.R. *In preparation*.**