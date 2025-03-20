from src.deep_similarity.dataloader import DataLoader
from src.deep_similarity.deep_similarity import DeepSimilarity
import os
from torchvision import transforms
import torch
import numpy as np


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config", "config.yaml")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    model = DeepSimilarity()

    stereo_dataset = DataLoader(config_path, mode="stereo", transform=transform)
    stereo_loader = torch.utils.data.DataLoader(stereo_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    timestep_dataset = DataLoader(config_path, mode="timestep", transform=transform)
    timestep_loader = torch.utils.data.DataLoader(timestep_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    stereo_scores = {
        'cosine': [],
        'euclidean': [],
        'dot_product': []
    }
    timestep_scores = {
        'cosine': [],
        'euclidean': [],
        'dot_product': []
    }
    
    model.eval()
    with torch.no_grad():
        for batch in stereo_loader:
            cosine_sim, euclidean_sim, dot_product_sim = model(batch)
            stereo_scores['cosine'].append(cosine_sim.cpu().numpy())
            stereo_scores['euclidean'].append(euclidean_sim.cpu().numpy())
            stereo_scores['dot_product'].append(dot_product_sim.cpu().numpy()) 
        for batch in timestep_loader:
            cosine_sim, euclidean_sim, dot_product_sim = model(batch)
            timestep_scores['cosine'].append(cosine_sim.cpu().numpy())
            timestep_scores['euclidean'].append(euclidean_sim.cpu().numpy())
            timestep_scores['dot_product'].append(dot_product_sim.cpu().numpy())
        for metric in ['cosine', 'euclidean', 'dot_product']:
            stereo_scores[metric] = np.concatenate(stereo_scores[metric])
            timestep_scores[metric] = np.concatenate(timestep_scores[metric])
    

    print("Printing average similarity scores\n")
    for metric in ['cosine', 'euclidean', 'dot_product']:
        stereo_mean = np.mean(stereo_scores[metric])
        timestep_mean = np.mean(timestep_scores[metric])
        print(f"{metric} Similarity:")
        print(f"Stereo pairs: {stereo_mean:.2f}")
        print(f"timestep pairs: {timestep_mean:.2f}")
        print(f"Difference: {abs(stereo_mean - timestep_mean):.2f}")
        print()


if __name__ == "__main__":
    main()
