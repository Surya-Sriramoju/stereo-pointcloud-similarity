# Stereo PointCloud & Deep Similarity

This project demonstrates two main tasks:

1. **Point Cloud Generation:**  
   Converts rectified stereo disparity maps and left stereo images into 3D point clouds using camera intrinsic parameters and filtering methods.

2. **Deep Similarity Evaluation:**  
   Computes similarity scores (cosine, Euclidean, and dot product) between image pairs (either left/right stereo pairs or consecutive left images) using a deep feature extractor (ResNet18).

### Setup
1. **Create and Activate a Virtual Environment:**

```bash
python3 -m venv venv
source venv/bin/activate
```
2. **Clone the Repository:**
```bash
git clone https://github.com/Surya-Sriramoju/stereo-pointcloud-similarity.git
cd stereo-pointcloud-similairty
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Place the datasets (left_stereo, right_stereo, disparity) inside a directory called dataset (or adjust the paths in config/config.yaml accordingly).

5. All parameters (camera intrinsics, maximum depth, disparity thresholds, filter settings, and output paths) can be adjusted in the config/config.yaml file.

### Running the code
1. **Point cloud generation:**
Run the following command from the project root. If the save flag is set to True in the configuration, the generated point cloud files will be saved in the dataset/output directory in .pcd format. Otherwise, a random point cloud from the dataset will be visualized. There are 3 main scripts for point cloud generation in the src folder
- generate_point_cloud.py: 
    - Extracts necessary information from the config file (e.g., camera parameters, max depth, filtering options).
    - Generates a 3D point cloud from disparity maps using geometry
- dataloader.py: 
    - Handles data loading efficiently, making it easier to iterate over stereo images and disparity maps.
- filter.py: 
    - Implements statistical outlier removal to clean the point cloud by removing noise and spurious points.
    - Contains a thresholding function to mask out sky pixels and improve the quality of reconstruction.
```bash
python run_point_cloud.py
```
2. **Deep Similarity:**
Run the following code from project root, the code compares cosine, euclidean and dot product similarities and saves a histogram image of the similarity score and frequency.
```bash
python run_similarity.py
```

## Technical details
1. **Point cloud generation**
- z = (f * b) / d

- x = (xl * z) / f

- y = (yl * z) / f

Where:
- z is the depth (distance from the cameras) found out using triangulation
- x,y are the 3D world coordinates
- f is the focal length
- b is the baseline
- d is the disparity (xl-xr)
- xl, yl are the pixel coordinates in the left image

Initially, the generated point cloud contained significant noise. To mitigate this, a statistical outlier removal method, similar to the k-nearest neighbors approach was applied. Additionally, noise from the sky was effectively removed using a thresholding technique.
- average time to process
    - Unfiltered point cloud = 0.16 sec
    - Filtered Point cloud = 1 sec

A sample image as follows

| **Unfiltered** | **Filtered** | **Filtered and sky masked** |
|:---------------------:|:---------------------:|:---------------------:|
| <img src="/samples/unfiltered.png" width="500"> | <img src="/samples/filtered.png" width="500"> | <img src="/samples/filtered_sky_removed.png" width="500"> |

2. Deep Similarity: 
This consists of a feature extractor, namely resnet 18, finally we receive a 256 dimensional vector, so basically 2 images pass through the network. Once we have the similarity score has been achieved, I have used following metrics.

-  Cosine Similarity: Cosine similarity checks how similar two things are by measuring the angle between their feature vectors. A score of 1 means they’re very similar, while -1 means they’re completely different.
    - average similarity
        - Stereo: 0.99
        - Timestep: 0.88
        - difference: 0.11

- Euclidean Distance metric: Euclidean distance measures how far apart two points are in space. A smaller distance means they’re more similar, while a larger distance means they’re different. To compute similarity, we take the reciprocal of the distance, so closer points have higher similarity scores.
    - average similarity
        - Stereo: 0.09
        - Timestep: 0.003
        - difference: 0.05

- Dot product similarity: Dot product similarity measures how aligned two feature vectors are. A higher value means the vectors point in a similar direction, indicating greater similarity. To keep the values in a useful range, we apply a sigmoid function.
    - average similarity
        - Stereo: 0.89
        - Timestep: 0.83
        - difference: 0.06

- Among all the metrics, cosine similarity stood out because it consistently highlighted meaningful relationships between image pairs, even when brightness or contrast varied. Unlike Euclidean distance, which is sensitive to absolute differences










