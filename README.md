# ConvSequential-SLAM

This repository provides the official implementation of ConvSequential-SLAM, introduced in the following paper: <br>

**Title:** ConvSequential-SLAM: A Sequence-Based, Training-Less Visual Place Recognition Technique for Changing Environments <br>
**Authors:** Mihnea-Alexandru Tomita, Mubariz Zaffar, Michael J. Milford, Klaus D. McDonald-Maier and Shoaib Ehsan

Published in IEEE Access, vol. 9, pp. 118673-118683, 2021 and available 📑 [here](https://doi.org/10.1109/ACCESS.2021.3107778).

## Overview

Visual Place Recognition (VPR) is the task of recognizing previously visited locations despite changes in viewpoint and appearance. Existing handcrafted methods often fail under strong appearance variations, while deep-learning approaches require heavy computation and extensive training.

**ConvSequential-SLAM** is a sequence-based, handcrafted VPR technique that achieves state-of-the-art performance under challenging conditions. The method leverages:
- **Convolutional matching** to improve robustness to moderate viewpoint variations.
- **Regional, block-normalized HOG descriptors** to achieve conditional invariance without relying on contrast-enhanced pixel matching.
- **Information-gain** analysis from consecutive query images to determine the minimum sequence length needed, tailored specifically for ConvSequential-SLAM.
- **Entropy-based salient region extraction** to dynamically adjust the sequence length based on the environment, rather than using a fixed length as in traditional sequence-based VPR techniques.

> Note: This repository also provides a version of ConvSequential-SLAM that uses a static (fixed) sequence length in addition to the dynamic sequence length version described above. We discuss both versions in more detail in the *Running ConvSequential-SLAM* section of this repo.

## Installation / Setup

**ConvSequential-SLAM** is implemented in *Python 2.7* and requires the following libraries to run:
`cv2`, `numpy`, `scikit-image`, `matplotlib`, `numba` and `scipy`.

### Installing Dependencies 
You can install the required Python packages using the following command: <br>
`pip2 install numpy opencv-python scikit-image matplotlib numba scipy`

 **Note:** Make sure you are using Python 2.7. Check your version with `python2 --version`
 
## Running ConvSequential-SLAM
This repository provides two versions of ConvSequential-SLAM:
1. `ConvSequential-SLAM.py` dynamically adapts the sequence length based on the environment.
2. `ConvSequential-SLAM_static_k.py` uses a fixed sequence length `k`, specified by the user. <br>

Both versions are ready to run out-of-the-box with the provided folder structure.

### Required User Modifications
Before running either version, you need to specify the following: <br>
<pre>
total_Query_Images = 100  # Number of query images in your dataset
total_Ref_Images = 100    # Number of reference images in your dataset

# Update these paths to point to your dataset
query_directory = '/home/mihnea/datasets/campus_loop_original/live/'
ref_directory = '/home/mihnea/datasets/campus_loop_original/memory/'

# Directory for visualizing entropy-based regions extraction
out_directory = '/home/mihnea/ConvSequential-SLAM/entropy_extracted_regions/'  
</pre>

### Running the Dynamic Version
Open a terminal in the `ConvSequential-SLAM/` folder then execute the main script using the following command:<br>
`python2 ConvSequential-SLAM.py`

**Default parameters used in our experiments:**
<pre>
W1 = H1 = 512
W2 = H2 = 16      # HOG cell-size
L = 8             # HOG bin size
ET = 0.5          # Entropy threshold (0-1)
IT = 0.9          # Overlapping threshold (0-1)
min_k = 1
max_k_IG = 15     # Corresponds to max_overlap in the code
max_k = 25
</pre>

> These values form the backbone of the system and are responsible for computing the optimal dynamic sequence length. To reproduce the results presented in the paper, these parameters should remain unchanged. For more information, please visit the paper.

### Running the Static Version 
Open a terminal in the `ConvSequential-SLAM/` folder then execute the main script using the following command: <br>
`python2 ConvSequential-SLAM_static_k.py`

**Important:** In addition to the changes specified in the *Required User Modifications* section, this version of ConvSequential-SLAM requires the sequence length `k` to be specified.

## Region-of-Interest (ROI) Extraction

<p>
ConvSequential-SLAM extracts salient regions from an image depending on the entropy threshold (ET). As ET increases, non-informative elements such as walls and floors are filtered out, resulting in fewer detected regions of interest (ROIs) per image. This is illustrated below.
</p>

<div align="center">
<table>
  <tr>
    <td><img src="path/to/image1.png" width="200"><br>Query Image</td>
    <td><img src="path/to/image2.png" width="200"><br>ET = 0.4</td>
    <td><img src="path/to/image3.png" width="200"><br>ET = 0.5</td>
  </tr>
  <tr>
    <td><img src="path/to/image4.png" width="200"><br>ET = 0.6</td>
    <td><img src="path/to/image5.png" width="200"><br>ET = 0.7</td>
    <td><img src="path/to/image6.png" width="200"><br>ET = 0.8</td>
  </tr>
</table>
</div>




