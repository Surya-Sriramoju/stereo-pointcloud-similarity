import argparse
import os
import yaml
import open3d as o3d
from src.point_cloud.dataloader import DataLoader
from src.point_cloud.generate_point_cloud import GeneratePoints
import random
from tqdm import tqdm
from time import time

def main():
    """
    If the save attribute is true, it will save the point clouds in dataset/output directory
    If the save attraibute is false, it will randomly visualize a 3D pointcloud from the set of 100 images
    """
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    dataloader = DataLoader(config)
    pointcloud = GeneratePoints(config)

    count = 0
    output_dir = os.path.join(config['paths']['data_root'], 'output')
    os.makedirs(output_dir, exist_ok=True)

    #saves the point clouds if the save flag is true in config.yaml
    if config['output']['save']:
        print("Generating and saving point clouds")
        for sample in tqdm(dataloader, desc="Processing point clouds"):
            disp, img = sample
            pcd = pointcloud.point_cloud(disp, img)
            filename = f"pointcloud_{count:03d}.pcd"
            save_path = os.path.join(output_dir, filename)
            o3d.io.write_point_cloud(save_path, pcd)
            count += 1
        #visualizes a random point cloud 
    else:
        number = random.randint(0, 99)
        disp, img = dataloader[number]
        pcd = pointcloud.point_cloud(disp, img)
        o3d.visualization.draw_geometries([pcd],
                                 zoom=0.3412,
                                 front=[0.4257, -0.2125, -0.8795],
                                 lookat=[2.6172, 2.0475, 1.532],
                                 up=[-0.0694, -0.9768, 0.2024])

if __name__ == "__main__":
    main()