from typing import Dict
import torch
import logging

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere

logger = logging.getLogger()

class MultiTrainer(BasicTrainer):
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)
        self.render_each_class = True
        
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes.keys():
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            elif class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)
            elif class_name == 'Ground':
                model = import_str(model_cfg.type)().to(self.device)
            else:
                raise Exception("Not supported class name: {}".format(class_name))
            self.models[class_name] = model
            
        logger.info(f"Initialized models: {self.models.keys()}")
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )
            return False
        else:
            return True

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:

        # Ground network initialization
        if 'Ground' in self.models.keys():
            self.omnire_w2neus_w = self.models['Ground'].omnire_w2neus_w.to(dataset.pixel_source.camera_data[0].cam_to_worlds.device)
            self.cam_0_to_neus_world = self.omnire_w2neus_w @ dataset.pixel_source.camera_data[0].cam_to_worlds
            self.ego_points = self.cam_0_to_neus_world[:, :3, 3]
            self.ego_points[:, 2] -= 1.51
            self.ego_normals = self.cam_0_to_neus_world[:, :3, :3] @ torch.tensor([0, -1, 0]).type(torch.float32).to(self.cam_0_to_neus_world.device)
            self.models['Ground'].ego_points = self.ego_points
            self.models['Ground'].ego_normals = self.ego_normals
            self.models['Ground'].pretrain_sdf()
            
            self.models['Ground'].validate_mesh()
        # get instance points
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict = {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type='RigidNodes',
                **self.model_config["RigidNodes"]["init"]
            )

        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type='DeformableNodes',        
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"]
            )

        if "SMPLNodes" in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )
        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict, **smplnode_pts_dict}
        
        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = [] 
        
        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            
            empty = False
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None
                
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)
                
                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict=allnode_pts_dict
                )
                
                model.create_from_pcd(
                    init_means=processed_init_pts["pts"], init_colors=processed_init_pts["colors"]
                )
                
            if class_name == 'RigidNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=rigidnode_pts_dict
                )
                
            if class_name == 'DeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=deformnode_pts_dict
                )
            
            if class_name == 'SMPLNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smplnode_pts_dict
                )
                
            if empty:
                empty_classes.append(class_name)
                logger.warning(f"No points for {class_name} found, will remove the model")
            else:
                logger.info(f"Initialized {class_name} gaussians")
        
        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")
                
        logger.info(f"Initialized gaussians from pcd")
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
            
        image_infos: {
            'origins': torch.Tensor, [900 / d, 1600 / d, 3]. 都是同一个origin
            'viewdirs': torch.Tensor, [900 / d, 1600 / d, 3]. 
            'direction_norm': torch.Tensor, [900 / d, 1600 / d, 1]. ???
            'pixel_coords': torch.Tensor, [900 / d, 1600 / d, 2]. normalized pixel coordinates
            'normed_time': torch.Tensor, [900 / d, 1600 / d]. normalized time. 猜测是整个bag的时间戳在0-1之间的归一化
            'img_idx': torch.Tensor, [900 / d, 1600 / d]. 
            'frame_idx': torch.Tensor, [900 / d, 1600 / d].
            'pixels': torch.Tensor, [900 / d, 1600 / d, 3]. RGB
            'sky_masks': torch.Tensor, [900 / d, 1600 / d]. 估计1代表天空
            'dynamic_masks': torch.Tensor, [900 / d, 1600 / d]. 
            'human_masks': torch.Tensor, [900 / d, 1600 / d].
            'vehicle_masks': torch.Tensor, [900 / d, 1600 / d].
            'lidar_depth_map': torch.Tensor, [900 / d, 1600 / d].
        }
        
        camera_infos: {
            'cam_id': torch.Tensor, [900 / d, 1600 / d].
            'cam_name': str.
            'camera_to_world': torch.Tensor, [4, 4]. #TODO: nuscenes相机高度从哪里来
            'height': torch.Tensor, [1]. image height
            'width': torch.Tensor, [1]. image width
            'intrinsics': torch.Tensor, [3, 3].
        }
        
        
        self.models: dict_keys(['Background', 'RigidNodes', 'DeformableNodes', 'SMPLNodes', 'Sky', 'Affine', 'CamPose'])

        self.gaussian_classes: dict_keys(['Background', 'RigidNodes', 'SMPLNodes', 'DeformableNodes'])
        """

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        
        # prapare data
        processed_cam = self.process_camera(    # 如果要对pose优化，或者perturb（TODO 为什么），在这里处理
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(    # 从各个gaussian model中收集gaussian，每个gaussian model都有get_gaussians接口，根据相机参数获得gaussian
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        ) # gs: dataclass_gs(_means, _scales, _quats, _rgbs, _opacities)
        

        # render gaussians
        # outputs: 
        #   rgb_gaussians: torch.Tensor, [900 / d, 1600 / d, 3]. 
        #   depth: torch.Tensor, [900 / d, 1600 / d, 1].
        #   opacity: torch.Tensor, [900 / d, 1600 / d, 1].
        outputs, render_fn = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        if 'Ground' in self.models.keys():
            # render ground
            ground_model = self.models['Ground']
            rgb_ground = ground_model(image_infos, camera_infos)
            outputs['ground'] = rgb_ground
            
            # affine transformation # 对每个image_idx的rgb进行一次affine transform
            outputs["rgb"] = self.affine_transformation(
                outputs["rgb_gaussians"] + (rgb_ground['rgb_full'] + image_infos['sky_masks'].unsqueeze(-1) * outputs["rgb_sky"]) * (1.0 - outputs["opacity"]), image_infos
            )
            if self.models['Ground'].iter_step % 100 == 0:
                with torch.no_grad():
                    road_opacity_mean = outputs['opacity'][image_infos['road_masks'] == 1].mean()
                    sky_opacity_mean = outputs['opacity'][image_infos['sky_masks'] == 1].mean()
                    print(road_opacity_mean.item(), sky_opacity_mean.item())
                
                image_infos["gaussian_rgb"] = outputs["rgb"].detach()
                self.models['Ground'].validate_image(image_infos, camera_infos)
            if self.models['Ground'].iter_step % 1000 == 0:
                self.models['Ground'].validate_mesh()
        else:
            outputs["rgb"] = self.affine_transformation(
                outputs["rgb_gaussians"] + image_infos['sky_masks'].unsqueeze(-1) * outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
            )
        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                    outputs[class_name+"_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                    outputs[class_name+"_opacity"] = sep_opacity
                    outputs[class_name+"_depth"] = sep_depth

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth
        
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict