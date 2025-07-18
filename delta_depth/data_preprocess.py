# this file should prepare data required for training the policy.
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

# MODIFICATION: rull out depth images and grip_pct.
# MODIFICATION: this version should use only xyz in delta space, removing normalization.

device = 'cpu'
# device = "cuda" if torch.cuda.is_available() else "cpu"


# Parameters
pred_horizon = 17
obs_horizon = 2
action_horizon = 8

#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# pred_horizon = pred_horizon + 1 # since we are in delta space, we will lose one value for computation, so we manually add one.


class CustomDataset(Dataset):
    def __init__(self, episodes_dir,transform_rgb, transform_depth, obs_horizon=2,
                 pred_horizon=16, normalizer=None):
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.transform_rgb = transform_rgb

        self.transform_depth = transform_depth

        self.normalizer = normalizer
        self.samples = [] # should store all valid sequences

        for episode_name in os.listdir(episodes_dir):
            episode_path = os.path.join(episodes_dir, episode_name)
            if not os.path.isdir(episode_path): continue
            
            csv_path = os.path.join(episode_path, "episode.csv")
            if not os.path.exists(csv_path): continue
            dataframe = pd.read_csv(csv_path)

            # Get all time aligned-data
            episode_data = []
            for idx, row in dataframe.iterrows():
                row = row.tolist()
                timestamp = row[0]
                # proprio = row[1:8] + [grip_signal]  # (x, y, z, qw, qx, qy, qz, gripper)
                proprio = row[1:4] # (x, y, z, qw, qx, qy, qz)
                rgb_path = os.path.join(episode_path, "images", row[9])
                depth_path = os.path.join(episode_path, "images", row[10])

                episode_data.append({
                    "proprio": proprio,
                    "rgb": rgb_path,
                    "depth": depth_path
                })       

            # Arrange for sequences, containing observation sequences and prediction horizon
            # (Action horizon should be a subset of prediction horizon executed during inference time)
            total_length = len(episode_data)
            for i in range(total_length - pred_horizon): # Assuming pred_horizon > pred_horizon all the time
                obs_seq = episode_data[i:i+obs_horizon]
                act_seq = episode_data[i:i+pred_horizon]
                self.samples.append((obs_seq, act_seq))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx:int) -> dict[str, torch.Tensor]:
        obs_seq, act_seq = self.samples[idx]
        rgb_seq, depth_seq, proprio_seq, action_seq = [], [], [], []

        for frame in obs_seq:
            rgb = Image.open(frame["rgb"]).convert("RGB")
            depth = Image.open(frame["depth"]).convert("L")

            rgb = self.transform_rgb(rgb)
            depth = self.transform_depth(depth)

            rgb_seq.append(rgb)
            depth_seq.append(depth)
            proprio = torch.tensor(frame["proprio"], dtype=torch.float32)
            # normalize proprioception in the observation
            if self.normalizer:
                proprio = self.normalizer.normalize(proprio)
                # append the proprioception to observation horizon
            proprio_seq.append(proprio)

        # for frame in act_seq:
        #     action = torch.tensor(frame["proprio"], dtype=torch.float32)
        #     # normalize the action in the prediction horizon
        #     if self.normalizer:
        #         action = self.normalizer.normalize(action)
        #         # append the action to the predicted action horizon
        #     action_seq.append(action)


        # --- Delta action computation ---
        for i in range(len(act_seq) - 1):
            pos_curr = torch.tensor(act_seq[i + 1]["proprio"], dtype=torch.float32)
            pos_prev = torch.tensor(act_seq[i]["proprio"], dtype=torch.float32)
            delta = pos_curr - pos_prev  # delta(x, y, z)
            action_seq.append(delta)

        return {
            "rgb": torch.stack(rgb_seq),            # (obs_horizon, 3, H, W)
            "depth": torch.stack(depth_seq),        # (obs_horizon, 1, H, W)
            "proprio": torch.stack(proprio_seq),    # (obs_horizon, 8) conditioned actions, input
            "action": torch.stack(action_seq)       # (pred_horizon, 8) predicted actions
        }
    

class LinearNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps=1e-8):
        self.mean = mean  # shape (3,) for position
        self.std = std    # shape (3,) for position
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        pos = (x[..., :3] - self.mean) / (self.std + self.eps)  # normalize x, y, z
        quat = x[..., 3:]  # leave quaternion unchanged
        return pos
        return torch.cat([pos, quat], dim=-1)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[..., :3] * (self.std + self.eps) + self.mean  # unnormalize x, y, z
        quat = x[..., 3:]  # leave quaternion unchanged
        return pos
        return torch.cat([pos, quat], dim=-1)

    def normalize_to_device(self, x: torch.Tensor) -> torch.Tensor:
        pos = (x[..., :3] - self.mean.to(device)) / (self.std.to(device) + self.eps)
        quat = x[..., 3:]
        return pos
        return torch.cat([pos, quat], dim=-1)

    def unnormalize_to_device(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[..., :3] * (self.std.to(device) + self.eps) + self.mean.to(device)
        quat = x[..., 3:]
        return pos
        return torch.cat([pos, quat], dim=-1)

    @classmethod
    def from_dataset(cls, dataset: torch.utils.data.Dataset):
        all_positions = []
        for i in range(len(dataset)):
            actions = dataset[i]['action']
            all_positions.append(actions[:, :3])  # only x, y, z


        all_positions = torch.cat(all_positions, dim=0)  # (N, 3)
        mean = all_positions.mean(dim=0)
        std = all_positions.std(dim=0)
        return cls(mean, std)



path_name = "../DataCollection"

# Transformations for RGB images and depth images
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # converts to [0, 1] range
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

# remove depth images.
# transform_depth = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # (1, H, W), values in [0, 1]
#     transforms.Normalize(mean=[0.5], std=[0.5])  # simple normalization
# ])


transform_depth = transforms.Compose([
    transforms.Lambda(lambda x: np.array(x).astype(np.float32) / 1000.0),  # mm → meters
    transforms.Lambda(lambda x: np.clip(x, 0.2, 3.0)),                     # clamp to [0.2, 3.0]
    transforms.Lambda(lambda x: 2 * (x - 0.2) / (3.0 - 0.2) - 1),          # normalize to [-1, 1]
    transforms.Lambda(lambda x: Image.fromarray(x)),                      # back to PIL
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # shape: (1, H, W)
])




# load raw dataset
raw_dataset = CustomDataset(
    episodes_dir=path_name,
    transform_rgb=transform_rgb,
    transform_depth=transform_depth,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    normalizer=None
)

# leave below lines commented, since there is no need to calcualte mean, std again.
# get mean and std using classmethod of LinearNormalizer
# normalizer = LinearNormalizer.from_dataset(raw_dataset) # get mean and std

# torch.save({
#     'mean': normalizer.mean,
#     'std': normalizer.std
# }, 'normalizer_stats.pth')







