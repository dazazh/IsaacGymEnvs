from isaacgym import gymapi,gymtorch
import math
import random
import numpy as np
import cv2
import torch

gym = gymapi.acquire_gym()

# get default set of parameters
sim_params = gymapi.SimParams()
# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
# set Flex-specific parameters
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5
# create sim with these parameters
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

fr5_asset_root = "assets"
fr5_asset_file = "urdf/fr5model/urdf/fr5model.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
fr5_asset = gym.load_asset(sim, fr5_asset_root, fr5_asset_file, asset_options)
print(gym.get_asset_dof_properties(fr5_asset))
print("test test")

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
# create the ground plane
gym.add_ground(sim, plane_params)

camera_asset = gym.create_box(sim,0.1,0.2,0.1,gymapi.AssetOptions())

table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

# set up the env grid
num_envs = 64
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
fr5_actor_handles = []
table_actor_handles = []
box_actor_handles = []
camera_handles = []
# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    fr5_pose = gymapi.Transform() # p represents position
    fr5_pose.p = gymapi.Vec3(0, 0, 0.0)
    fr5_actor_handle = gym.create_actor(env, fr5_asset, fr5_pose, "fr5", i, 1)
    fr5_actor_handles.append(fr5_actor_handle)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)
    table_actor_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    table_actor_handles.append(table_actor_handle)

    box_pose = gymapi.Transform()
    box_pose.p.x = table_pose.p.x + random.uniform(-0.2, 0.1)
    box_pose.p.y = table_pose.p.y + random.uniform(-0.3, 0.3)
    box_pose.p.z = table_dims.z + 0.5 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), random.uniform(-math.pi, math.pi)) #设定随机初始旋转角度
    box_actor_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    box_actor_handles.append(box_actor_handle)

    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = 512
    camera_props.height = 512
    camera_props.use_collision_geometry = True
    camera_handle = gym.create_camera_sensor(env, camera_props)
    camera_handles.append(camera_handle)
    end_effector = gym.find_actor_rigid_body_handle(env, fr5_actor_handle,'Link6')
    camera_pos = [0, 0, -1]

    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(*camera_pos)
    gym.attach_camera_to_body(camera_handle, env, end_effector, local_transform, gymapi.FOLLOW_TRANSFORM)
    # camera_pos = [0, 0, 1]
    # local_transform = gymapi.Transform()
    # local_transform.p = gymapi.Vec3(*camera_pos)
    # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
    # gym.attach_camera_to_body(camera_handle, env, fr5_actor_handle, local_transform, gymapi.FOLLOW_TRANSFORM)


cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

# Point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + envs_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 8, 1)
dof_vel = dof_states[:, 1].view(num_envs, 8, 1)
pos_action = torch.ones_like(dof_pos).squeeze(-1)*math.pi/4
# effort_action = torch.ones_like(pos_action)
# print(dof_states)
# print(dof_pos)
# print(dof_vel)

# print(gymtorch.wrap_tensor(gym.acquire_jacobian_tensor(sim, "fr5")))
# print("test")
print(pos_action)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))


    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)
    camera_rgba_tensor = gym.get_camera_image_gpu_tensor(sim, envs[0], camera_handles[0],
                                                                        gymapi.IMAGE_COLOR)
    camera_depth_tensor = gym.get_camera_image_gpu_tensor(sim, envs[0], camera_handles[0],
                                                                        gymapi.IMAGE_DEPTH)
    torch_camera_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
    torch_camera_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
    rgba_img = torch_camera_rgba_tensor.clone().cpu().numpy()
    _depth_img = torch_camera_depth_tensor.clone().cpu().numpy()
    gym.end_access_image_tensors(sim)
    rgb_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)
    cv2.imshow('RGB Image', rgb_img)
    _depth_img[np.isinf(_depth_img)] = -256
    depth_img = cv2.normalize(_depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow('Depth Image', depth_img)
    cv2.waitKey(1)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
