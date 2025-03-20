import numpy as np
from .filter import OutlierFilter


class GeneratePoints():
    def __init__(self, data):
        """
        Reads the data from the config file necessary for point cloud generation
        and initialises it
        """
        try:
            self.data = data
            self.camera_params = self.data['camera']
            self.fx = self.camera_params['fx']
            self.fy = self.camera_params['fy']
            self.cx = self.camera_params['cx']
            self.cy = self.camera_params['cy']
            self.b = self.camera_params['b']

            self.max_depth = self.data['point_cloud']['depth_range']['max']
            self.disparity_thresh = self.data['point_cloud']['disparity']['min_thresh']

            self.filter = self.data['point_cloud']['filter']
            self.roi = self.data['filter_methods']['threshold']['roi']
            
            self.sky_threshold = self.data['filter_methods']['threshold']['sky_threshold']
            self.thresh_min = self.data['filter_methods']['threshold']['thresh_min']
        except Exception as e:
            raise KeyError(f"Missing values or mismatch: {e}")

        # Initialized filter object to filter out the noise in the point cloud
        self.filter_class = OutlierFilter(data)
    
    def point_cloud(self, disparity_map: np.ndarray, left_image: np.ndarray):
        """
        Generates point clouds based on the disparity and camera parameters
        Input:
            Disparity Map
            Left Stereo Image
        Output:
            pcd: Point Cloud object
        """
        valid_mask = disparity_map > self.disparity_thresh
        if self.sky_threshold:
            thresh = self.filter_class.threshold(left_image)
            roi_mask = thresh[:self.roi, :] != 0
            valid_mask[:self.roi, :][roi_mask] = False

        y_indices, x_indices = np.where(valid_mask)
        z = (self.fx * self.b) / disparity_map[valid_mask]
        depth_mask = z<self.max_depth*1000
        z = z[depth_mask]
        y_indices = y_indices[depth_mask]
        x_indices = x_indices[depth_mask]
        x = ((x_indices - self.cx) * z) / self.fx
        y = ((y_indices - self.cy) * z) / self.fy
        colors = left_image[y_indices, x_indices] / 255.0
        points = np.column_stack((x,y,z))
        
        if self.filter and len(points)>1000:
            pcd = self.filter_class.statistical_outlier_removal(points, colors)
        else:
            pcd = self.filter_class.pcd_format(points, colors)
        return pcd

