import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MultiModalNet(nn.Module):
    def __init__(self, embedding_dim=128, meta_dim=16, backbone_name='tf_efficientnet_b3_ns'):
        super(MultiModalNet, self).__init__()
        
        # 1. Image Branch
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.img_fc = nn.Linear(self.backbone.num_features, embedding_dim)
        
        # 2. Metadata Branch (Lat, Lon, Timestamp)
        # Input: [lat, lon, timestamp] (These should be normalized/scaled)
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, meta_dim)
        )
        
        # 3. Fusion
        self.final_fc = nn.Linear(embedding_dim + meta_dim, embedding_dim)
        
    def forward(self, img, meta):
        # Image feature
        x_img = self.backbone(img)
        x_img = self.img_fc(x_img)
        x_img = F.normalize(x_img, p=2, dim=1)
        
        # Meta feature
        x_meta = self.meta_fc(meta)
        x_meta = F.normalize(x_meta, p=2, dim=1)
        
        # Concatenate
        x_combined = torch.cat((x_img, x_meta), dim=1)
        
        # Final embedding
        x_out = self.final_fc(x_combined)
        x_out = F.normalize(x_out, p=2, dim=1)
        
        return x_out