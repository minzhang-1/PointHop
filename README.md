# PointHop: *An Explainable Machine Learning Method for Point Cloud Classification*
Created by Min Zhang, Haoxuan You, Pranav Kadam, Shan Liu, C.-C. Jay Kuo from University of Southern California.

![introduction](https://github.com/minzhang-1/PointHop/blob/master/doc/intro.png)

### Introduction
This work is an official implementation of our [arXiv tech report](https://arxiv.org/abs/1907.12766). We proposed a novel explainable machine learning method for point cloud, called the PointHop method.

We address the problem of unordered point cloud data using a space partitioning procedure and developing a robust descriptor that characterizes the relationship between a point and its one-hop neighbor in a PointHop unit. Furthermore, we used the Saab transform to reduce the attribute dimension in each PointHop unit. In the classification stage, we fed the feature vector to a classifier and explored ensemble methods to improve the classification performance. It was shown by experimental results that the training complexity of the PointHop method is significantly lower than that of state-of-the-art deep learning-based methods with comparable classification performance. 

In this repository, we release code and data for training a PointHop classification network on point clouds sampled from 3D shapes.

### Citation
If you find our work useful in your research, please consider citing:

	@article{zhang2019pointhop,
	  title={PointHop: An Explainable Machine Learning Method for Point Cloud Classification},
	  author={Zhang, Min and You, Haoxuan and Kadam, Pranav and Liu, Shan and Kuo, C-C Jay},
	  journal={arXiv preprint arXiv:1907.12766},
	  year={2019}
	}

### Installation

The code has been tested with Python 3.5. You may need to install h5py, pytorch, sklearn, pickle and threading packages.

To install h5py for Python:
```bash
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

### Usage
To train a single model to classify point clouds sampled from 3D shapes:

    python3 train.py

After the above training, we can evaluate the single model.

    python3 evaluate.py

If you would like to achieve better performance, you can change the argument `ensemble` from `False` to `True` in both `train.py` and `evaluate.py`.

If you run the code on your laptop with small memory, you can change the argument `num_batch_train` or `num_batch_test` larger. To get the same speed as the paper, set `num_batch_train` as 1 and `num_batch_test` as 1.

Log files and network parameters will be saved to `log` folder. Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.


