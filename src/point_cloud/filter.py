import open3d as o3d
import numpy as np
import cv2

class OutlierFilter():
    def __init__(self, data):
        """
        Initialises the OutlierFilter, this class is basically used to add methods in the future
        or select the best method for outlier removal.
        """
        self.data = data
        self.filter_type = self.data['filter_methods']['statistical_outlier_removal']
        self.nb_neighbors = int(self.filter_type['nb_neighbors'])
        self.std_ratio = float(self.filter_type['std_ratio'])
        self.thresh_filter = self.data['filter_methods']['threshold']
        self.roi = self.thresh_filter['roi']
        self.thresh_min = self.thresh_filter['thresh_min']
        self.thresh_max = self.thresh_filter['thresh_max']

    def statistical_outlier_removal(self, points: np.ndarray, colors: np.ndarray):
        """
        Removes outliers and cleans the noise in the point cloud
        Input:
            points: numpy array of 3D points in space
            colors: Colors associated to the 3D points
        Output:
            pcd: Filtered pointcloud
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_filtered,_ = pcd.remove_statistical_outlier(nb_neighbors = self.nb_neighbors, std_ratio = self.std_ratio)
        return pcd_filtered

    def threshold(self, image: np.ndarray):
        """
        Masks the sky avoiding noise in the pointcloud
        Input:
            image: image of the scene
        Output:
            Binary threshold of the image for filtering out the sky
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, self.thresh_min, self.thresh_max, cv2.THRESH_BINARY)
        return img


    def pcd_format(self, points: np.ndarray, colors: np.ndarray):
        """
        For converting the 3D points and colors to appropriate
        Point cloud format

        Input:
            points: numpy array of 3D points in space
            colors: Colors associated to the 3D points
        Output:
            pcd: Point cloud object
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd


