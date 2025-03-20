import torch
from torchvision.models import resnet18
from torchvision import transforms


class DeepSimilarity(torch.nn.Module):
    def __init__(self):
        super(DeepSimilarity, self).__init__()

        # Load a ResNet18 model pre-trained on ImageNet
        original_resnet18 = resnet18(pretrained=True)

        # Remove the last fully connected layer (fc) to use as a feature extractor
        self.feature_extractor = torch.nn.Sequential(*list(original_resnet18.children())[:-3])

    def forward(self, x):
        # x shape: (b, 2, c, h, w)
        b, pair, c, h, w = x.shape

        # Flatten the batch and pair dimension for processing through the CNN
        x = x.view(-1, c, h, w)  # Shape: (b*2, c, h, w)

        # Get the feature map from the CNN
        feature_map = self.feature_extractor(x)  # Shape: (b*2, 256, 14, 14)
        features = feature_map.reshape(b, pair, feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
        
        embedding1 = features[:, 0]
        embedding2 = features[:, 1]

        # Apply your similarity metric here
        # print(embedding1.flatten(1).shape)
        # cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(
            embedding1.flatten(1),
            embedding2.flatten(1),
            dim=1
        )
        cosine_similarity = (cosine_similarity + 1) / 2

        # euclidean similarity
        euclidean_distance = torch.sqrt(torch.sum((embedding1.flatten(1) - embedding2.flatten(1))**2, dim=1))

        euclidean_similarity = 1 / (1 + euclidean_distance)

        # dot product similarity

        dot_product_similarity = torch.sum(embedding1.flatten(1) * embedding2.flatten(1), dim=1)
        dot_product_similarity = torch.sigmoid(dot_product_similarity / 1000)


        return cosine_similarity, euclidean_similarity, dot_product_similarity

