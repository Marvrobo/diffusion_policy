# this file should train the model.
# this file should implement functions for testing and training the mode
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
# from model.model import ConditionalUnet1D, DepthCNN, get_resnet, replace_bn_with_gn
from model.model import ConditionalUnet1D, get_resnet, replace_bn_with_gn
import csv
from data_preprocess import CustomDataset, LinearNormalizer
from torchvision import transforms
"""
2. set hyperparameters
3. set directory, load dataset
4. set model name(how to save the model)
"""


# NOTICE Before training:
# recalculate mean and std for action.



log_path = "training_loss_log.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 1000
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
NUM_DIFFUSION_ITERS = 1000

path_name = "DataCollection"  # the path of dataset

# Transformations for RGB images and depth images
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # converts to [0, 1] range
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

# transform_depth = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # (1, H, W), values in [0, 1]
#     transforms.Normalize(mean=[0.5], std=[0.5])  # simple normalization
# ])


# load raw dataset
raw_dataset = CustomDataset(
    episodes_dir=path_name,
    transform_rgb=transform_rgb,
    # transform_depth=transform_depth,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    normalizer=None
)


# # get mean and std using classmethod of LinearNormalizer
# normalizer = LinearNormalizer.from_dataset(raw_dataset) # get mean and std

# Since we have already calculated mean, std; run the data preprocessor to store the normalizer.
stats = torch.load("normalizer_stats.pth")
normalizer = LinearNormalizer(stats["mean"], stats["std"])


norm_dataset = CustomDataset(
    episodes_dir=path_name,
    transform_rgb=transform_rgb,
    # transform_depth=transform_depth,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    normalizer=normalizer # normalize actions
)


dataloader = torch.utils.data.DataLoader(
    norm_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

# Construct ResNet-18 as RGB_vision_encoder:
rgb_vision_encoder = get_resnet('resnet18')


# Replace all BatchNorm with GroupNorm to work with EMA, otherwise performance will tank
rgb_vision_encoder = replace_bn_with_gn(rgb_vision_encoder)

rgb_vision_feature_dim = 512 # ResNet18 has output dim of 512
# obs_proprio_dim = 8 # xyz q(wxyz) grip_pct
obs_proprio_dim = 7 # x, y, z, qw, qx, qy, qz
# depth_vision_feature_dim = 512
# obs_dim = rgb_vision_feature_dim + obs_proprio_dim + depth_vision_feature_dim
obs_dim = rgb_vision_feature_dim + obs_proprio_dim
action_dim = 7

# create noise prediction network object
noise_pred_net = ConditionalUnet1D(
    input_dim = action_dim, 
    global_cond_dim=obs_dim * obs_horizon
)

nets = nn.ModuleDict({
    "rgb_encoder": rgb_vision_encoder,
    # "depth_encoder": depth_vision_encoder,
    "noise_pred_net": noise_pred_net
}).to(device)

# Exponential Moving Average, holding a copy of the model weights
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75
)

# Adam Optimizer, note that EMA parameters are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr = 1e-4, weight_decay=1e-6
)

# Consine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * NUM_EPOCHS
)

# DDPMScheduler with 100 diffusion iterations
noise_scheduler = DDPMScheduler(
    num_train_timesteps=NUM_DIFFUSION_ITERS,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=True, # after predciting the denoised sample, the ouput is clipped to the range[-1,1]
    prediction_type='epsilon'
)

with tqdm(range(NUM_EPOCHS), desc='Epoch') as tglobal:
    # epoch loop:    
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # RGB and depth images were normalized by transformations
                # actions were normalized by LinearNormalizer, notice that grip_pct left unchanged.
                nimage = nbatch['rgb'].to(device)
                # ndepth = nbatch['depth'].to(device)
                nagent_pos = nbatch['proprio'].to(device) # (B, obs_horizon, 7)
                naction = nbatch['action'].to(device)  # (B, obs_horizon, 7)
                B = naction.shape[0]

                # encoder vision features for RGB and Depth images.
                # "*" is a syntax sugar for unpacking a tuple.
                # RGB image
                RGB_image_features = nets["rgb_encoder"](nimage.flatten(end_dim=1)) 
                RGB_image_features = RGB_image_features.reshape(*nimage.shape[:2], -1)
                # [B, obs_horizon, C, H, W] -> [B * obs_horizon, C, H, W] -> [B * obs_horizon, D = 512] -> [B, obs_horizon, D = 512]
                

                # Depth image: (B, obs_horizon, 1, H, W) -> (B*obs_horizon, 1, H, W) -> (B * obs_horizon, D = 512)
                # depth_image_features = nets["depth_encoder"](ndepth.flatten(end_dim=1))
                # depth_image_features = depth_image_features.reshape(*ndepth.shape[:2], -1)
                #(B*obs_horizon, D) -> (B, obs_horizon, D)

                # Concatenate image feature, depth feature, and proprioperception feature
                # image_feature = torch.cat([RGB_image_features, depth_image_features], dim=-1) # image = rgb + depth

                image_feature = RGB_image_features # rule out depth image
                obs_feature = torch.cat([image_feature, nagent_pos], dim=-1) # obs = image + proprioception
                obs_cond = obs_feature.flatten(start_dim=1) # (B, obs_horizon * D_obs: (D_RGB + XXXD_depthXXX + D_proprio))

                # Sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)
                
                # Sample a diffusion iteration for each data point (*i.e.* what is the diffusion forwarding step for inputs)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B, ), device=device).long()

                # Forwarding diffusion process: add noise to the clean images according to noise_magnitude at each diffusion iteration(K)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)
                
                # Predict the noise according to previously generated timeteps, i.e. diffusion iterations
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond
                )

                # L2 Loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() # placed after optimizer.step() to prepare for the next iteration
                # Step LR every batch, since the LR schedule is based on toal number of steps.
                lr_scheduler.step() 

                # update Exponential Moving Average of the model weights
                # ema.step does not affect training, it maintains a moving average of the weights 
                # over training step. Typically used for inferecne, after training.
                ema.step(nets.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(los=loss_cpu)

            # Create a .csv file and write current loss with epoch so that we can plot the loss function later.
            avg_loss = np.mean(epoch_loss)

            # Write header if first epoch
            if epoch_idx == 0:
                with open(log_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "avg_loss"])

            # Append epoch loss
            with open(log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch_idx, avg_loss])

            # save the model every 100 epochs
            if epoch_idx % 100 == 0 and epoch_idx != 0:
                model_name = "diffusion_" + str(epoch_idx / 100) + ".pth"
                # Weights of the EMA model, used for inference.
                ema_nets = nets
                ema.copy_to(ema_nets.parameters())

                # Save the model
                torch.save(ema_nets.state_dict(), f"trained/{model_name}")



        tglobal.set_postfix(loss=np.mean(epoch_loss))







# the actions were also normalized. And action sequences should be 
# DENORMALIZE DURING INFERENCE TIME
