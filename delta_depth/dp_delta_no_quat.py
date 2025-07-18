# This file is the third version of diffusion policy.
# This version of diffusion policy: remove grip_pct, depth images; keep normalization for actions and RGB images
# (Should normalize action, then denormalize, during inference time)


import time
import torch
import bosdyn.client
from bosdyn.client.math_helpers import Quat, SE3Pose   
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.estop import EstopClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.power import PowerClient
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api import image_pb2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from data_preprocess import LinearNormalizer
from model.model import nets
import collections
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import io

# Notice. here if you use delta space. Then action here is actually delta values for xyz.

QUATERNION = (0.680577993, -0.000285911, 0.732674062, 0.001487711)

# This function cannot be used, for non-existing STATUS_AT_GOAL.
def send_and_wait_for_arm(command_client, cmd):
    cmd_id = command_client.robot_command(cmd)
    while True:
        feedback = command_client.robot_command_feedback(cmd_id)
        status = feedback.feedback.synchronized_feedback.arm_command_feedback.status
        if status == feedback.feedback.synchronized_feedback.arm_command_feedback.STATUS_AT_GOAL:
            print("Achieved target position")
            break
        time.sleep(0.1)

pred_horizon = 16
obs_horizon = 2
action_horizon = 8
action_dim = 3 # xyz qwxyz

# Use DDIM for less backwards steps.
noise_scheduler = DDIMScheduler(
    num_train_timesteps=25,
    clip_sample=True,
    prediction_type="epsilon"
)
print("Use DDIM for backwards process")

# Transforms
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # converts to [0, 1] range
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])


# RULE OUT DEPTH IMAGES.
# transform_depth = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # (1, H, W), values in [0, 1]
#     transforms.Normalize(mean=[0.5], std=[0.5])  # simple normalization
# ])

# print("pre-defined transformations for images normalization")

# Before inference, run the data preprocessor to store the normalizer.
stats = torch.load("normalizer_stats.pth")
normalizer = LinearNormalizer(stats["mean"], stats["std"])

ROBOT_IP = "192.168.80.3"
USERNAME = "admin"
PASSWORD = "pvwmr4j08osj"


# send all computation-related elements to the device.
device = 'cpu'
# device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_quaternion(qw, qx, qy, qz):
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if norm < 1e-6:  # avoid division by zero
        return 1.0, 0.0, 0.0, 0.0
    return qw / norm, qx / norm, qy / norm, qz / norm


# load the model then send it to GPU
grasp_model = nets()
# NOTICE: should modify the corrent model name.
grasp_model.load_state_dict(torch.load("trained/diffusion_9.0.pth"))
grasp_model = grasp_model.to(device)
print(grasp_model)

def get_observation(state_client, image_client):
        state = state_client.get_robot_state()
        snap = state.kinematic_state.transforms_snapshot
        body_T_hand = get_a_tform_b(snap, BODY_FRAME_NAME, HAND_FRAME_NAME)
        x, y, z = body_T_hand.x, body_T_hand.y,body_T_hand.z
        qw, qx, qy, qz = body_T_hand.rot.w, body_T_hand.rot.x, body_T_hand.rot.y, body_T_hand.rot.z
        # grip_close = 0 if state.manipulator_state.gripper_open_percentage > 90 else 1
        # proprio_tensor = torch.tensor([x, y, z, qw, qx, qy, qz, grip_close], dtype=torch.float32).to(device)
        # proprio_tensor = torch.tensor([x, y, z, qw, qx, qy, qz], dtype=torch.float32).to(device)
        proprio_tensor = torch.tensor([x, y, z], dtype=torch.float32).to(device)


        proprio_tensor = normalizer.normalize_to_device(proprio_tensor).unsqueeze(0) # (1, 8) -> (1, 7)

        # Get the real-time image.
        # NOTICE: SHOULD ALSO NORMALIZE THE IMAGES BEFORE SENDING TO THE MODEL.
        image_requests = [
            image_pb2.ImageRequest(
                image_source_name="hand_color_image",
                pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8
            ),
            image_pb2.ImageRequest(
                image_source_name="hand_depth_in_hand_color_frame",
                pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16
            )
        ]
        rgb_proto, depth_proto = image_client.get_image(image_requests)
        # rgb_proto = image_client.get_image(image_requests)


        h, w = rgb_proto.shot.image.rows, rgb_proto.shot.image.cols

        if rgb_proto.shot.image.format == image_pb2.Image.FORMAT_JPEG:
            rgb_np = np.array(Image.open(io.BytesIO(rgb_proto.shot.image.data)))
        else:
            h, w = rgb_proto.shot.image.rows, rgb_proto.shot.image.cols
            rgb_np = np.frombuffer(rgb_proto.shot.image.data, dtype=np.uint8).reshape((h, w, 3))


        # if depth_proto.shot.image.format == image_pb2.Image.FORMAT_JPEG:
        #     depth_np = np.array(Image.open(io.BytesIO(depth_proto.shot.image.data)))
        #     # This will load it as uint16 if the PNG stores 16-bit grayscale correctly.
        # elif depth_proto.shot.image.format == image_pb2.Image.FORMAT_RAW:
        #     h, w = depth_proto.shot.image.rows, depth_proto.shot.image.cols
        #     depth_np = np.frombuffer(depth_proto.shot.image.data, dtype=np.uint16).reshape((h, w))
        # else:
        #     raise ValueError(f"Unsupported depth image format: {depth_proto.shot.image.format}")
    
        # depth_np=depth_np.astype(np.float32)
        # rgb_np = np.frombuffer(rgb_proto.shot.image.data, dtype=np.uint8).reshape((h, w, 3))
        # depth_np = np.frombuffer(depth_proto.shot.image.data, dtype=np.uint16).reshape((h, w))

        rgb_img = Image.fromarray(rgb_np, mode="RGB")
        rgb_tensor = transform_rgb(rgb_img).unsqueeze(0).to(device)

        # depth_tensor = transform_depth(depth_np).unsqueeze(0).to('cpu')


        # depth_img = Image.fromarray(depth_np, mode="I;16")
        # depth_tensor = transform_depth(depth_img).unsqueeze(0).to(device)

        # synthesis observation (note that proprioceptions are considered as both input and conditioning) and append to deque.
        # obs = {"proprio": proprio_tensor, "rgb": rgb_tensor, "depth": depth_tensor}
        obs = {"proprio": proprio_tensor, "rgb": rgb_tensor}

        return obs

def main():

    sdk = bosdyn.client.create_standard_sdk("SpotMovementClient")
    robot = sdk.create_robot(ROBOT_IP)
    robot.authenticate(USERNAME, PASSWORD)
    robot.time_sync.wait_for_sync()
    estop_client = robot.ensure_client(EstopClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    power_client = robot.ensure_client(PowerClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    state_client = robot.ensure_client(RobotStateClient.default_service_name)
    lease = lease_client.take()
    lease_keep_alive = LeaseKeepAlive(lease_client)

    # Preparation: probably try manually at first:
    # unstow the arm, point the gripper towards the ground.
    print("Make sure the gripper is pointing towards the ground and be able to see the object")

    # Initialization: set up the initialization status: x, y, z, qw, qx, qy, qz, make the gripper open.
    x, y, z, qw, qx, qy, qz = 0.75924414, -0.121446341, 0.091800719, 0.729167342, 0.082954742, 0.67351222, -0.088401094
    move_cmd = RobotCommandBuilder.arm_pose_command(x=x,y=y,z=z,qw=qw,qx=qx,qy=qy,qz=qz,frame_name="body",seconds=2)
    command_client.robot_command(move_cmd)
    gripper_cmd = RobotCommandBuilder.claw_gripper_open_command()
    command_client.robot_command(gripper_cmd)
    time.sleep(3.0)



    # Maintain a global deque storing observations.
    obs_deque = collections.deque(maxlen=obs_horizon) # {"proprio", "image", "depth"}
    
    # LOOP: Manipulate the robot based on model predictions until the gripper close.
    # Each while iteration should correspond to one timestamp, Run the model at 2Hz
    grasp_success = False
    step_idx = 0 # used when the robot is executing predicted actions.
    while not grasp_success:
        B = 1 # during inference, match the input shape of the model, let B = 1

        obs = get_observation(state_client, image_client)
        obs_deque.append(obs)

        # Especially uring initialization, if there are no exactly two observations, then continue.
        if len(obs_deque) < 2:
            continue # wait until at least two observations were appended.

        # Already two observations, infer.
        with torch.no_grad():
            B = 1
            # Stack observations from deque: shape (obs_horizon, C, H, W) → (1, obs_horizon, C, H, W)
            rgb_batch = torch.stack([x["rgb"].squeeze(0) for x in obs_deque], dim=0).unsqueeze(0).to(device)
            # depth_batch = torch.stack([x["depth"].squeeze(0) for x in obs_deque], dim=0).unsqueeze(0).to(device)
            proprio_batch = torch.cat([x["proprio"] for x in obs_deque], dim=0).unsqueeze(0).to(device)  # (1, obs_horizon, 7)

            # Flatten to (B * obs_horizon, C, H, W)
            rgb_feat = grasp_model["rgb_encoder"](rgb_batch.flatten(end_dim=1))        # → (B * obs_horizon, D)
            # depth_feat = grasp_model["depth_encoder"](depth_batch.flatten(end_dim=1))  # → (B * obs_horizon, D)

            # Reshape back to (B, obs_horizon, D)
            rgb_feat = rgb_feat.view(B, obs_horizon, -1)
            # depth_feat = depth_feat.view(B, obs_horizon, -1)

            # Concatenate image features and proprio features
            # image_feat = torch.cat([rgb_feat, depth_feat], dim=-1)     # (B, obs_horizon, D+D)
            image_feat = rgb_feat
            obs_feat = torch.cat([image_feat, proprio_batch], dim=-1)  # (B, obs_horizon, D+D+8) -> (B, obs_horizon, D + 7)
            obs_cond = obs_feat.flatten(start_dim=1)                   # (B, obs_horizon * total_dim: (D + 7))



            # initialize action from pure Guassian Noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_inference_steps=25)

            # predict noise
            for k in noise_scheduler.timesteps:
                noise_pred = grasp_model["noise_pred_net"](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                ).to(device)

                # denoise (inverse diffusion step)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample # prev_sample: the denoised sample at the previous time step.
        
        # unnormalize action; naction: (B=1, pred_horizon, action_dim)
        # detach: all computations were performed on GPU in a computational graph. We need to detach it before send it to CPU.
        # action_pred = normalizer.unnormalize(naction.detach().cpu()[0]) # (pred_horizon, 8)
        action_pred = normalizer.unnormalize_to_device(naction[0]) # (pred_horizon, 7)



        # only take action_horizon number of actions
        start = obs_horizon - 1 # the first action should match the last observation
        end = start + action_horizon
        action = action_pred[start:end, :] # (action_horizon, action_dim)

        # execute action_horizon number of steps, appending observations along the way
        # running time 2Hz

        for i in range(len(action)):
             obs = get_observation(state_client, image_client)
             obs_deque.append(obs)

             # execute action
            #  x, y, z, qw, qx, qy, qz, grip_close = action[i]
             dx, dy, dz= action[i]
             x, y, z = x + dx, y + dy, z + dz
             qw, qx, qy, qz = QUATERNION

             qw, qx, qy, qz = normalize_quaternion(qw, qx, qy, qz)
             print(action[i])
             move_cmd = RobotCommandBuilder.arm_pose_command(x=x,y=y,z=z,qw=qw,qx=qx,qy=qy,qz=qz,
             frame_name="body",
             seconds=1
             )
            #  pose = SE3Pose(position=(x, y, z), rot=Quat(w=qw, x=qx, y=qy, z=qz))
            #  move_cmd = RobotCommandBuilder.arm_pose_command(
            #  root_frame_name="body",
            #  pos=pose.position,
            #  rot=pose.rotation,
            #  duration=0.5
            #  )

            # send the command to spot, and wait until finished
            #  send_and_wait_for_arm(command_client, move_cmd)

             command_client.robot_command(move_cmd)
             time.sleep(1.0)

             # if the grip_close is 1, command the gripper to close, exit the loop.
             # NOTICE THAT: HERE I SET THE THRESHOLD TO BE 0.7, instead of 0.5.
            #  if grip_close >= 0.7:
            #     gripper_cmd = RobotCommandBuilder.claw_gripper_close_command()
            #     command_client.robot_command(gripper_cmd)
            #     print("Gripper is closing...")
            #     time.sleep(1.0)
            #     grasp_success = True # exit the while loop
            #     break; # after exit the for loop
            #  time.sleep(3.0)

        state = state_client.get_robot_state()
        gripper_load = state.manipulator_state.is_gripper_holding_item
        if gripper_load:
            print("Successfully grasp the object, Raising arm...")
            snap = state.kinematic_state.transforms_snapshot
            body_T_hand = get_a_tform_b(snap, BODY_FRAME_NAME, HAND_FRAME_NAME)

            # Get current hand pose
            x, y, z = body_T_hand.x, body_T_hand.y, body_T_hand.z
            qw, qx, qy, qz = body_T_hand.rot.w, body_T_hand.rot.x, body_T_hand.rot.y, body_T_hand.rot.z

            # Raise 30 cm in Z
            # lift_pose = SE3Pose(position=(x, y, z + 0.30), rot=Quat(w=qw, x=qx, y=qy, z=qz))
            lift_cmd = RobotCommandBuilder.arm_pose_command(x, y, z, qw, qx, qy, qz, frame_name="body")
            command_client.robot_command(lift_cmd)

if __name__ == "__main__":
    print("getting into main...")
    main()
    