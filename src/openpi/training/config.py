"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.policies.s1_policy as s1_policy

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None
    


@dataclasses.dataclass(frozen=True)
class DataConfig: # 数据配置类
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = True

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False

    multi_rerobot: bool = False
    dataset_root: str | None = None
    


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                # return _transforms.Group(
                #     inputs=[
                #         _transforms.InjectDefaultPrompt(self.default_prompt),
                #         _transforms.ResizeImages(224, 224),
                #         _transforms.TokenizePrompt(
                #             _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                #             discrete_state_input=model_config.discrete_state_input,
                #         ),
                #         _transforms.PadStatesAndActions(model_config.action_dim),
                #     ],
                # )
                return _transforms.Group(
                    inputs=[
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt( # 分词处理器
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace( # 创建一个新的数据类实例，并更新指定的字段值
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class LeRobotS1DataConfig(DataConfigFactory):
    use_so3: list[bool] = dataclasses.field(default_factory=lambda: [False, False])
    repo_id: str|list[str] = "lerobot_so3_data_30hz"
    norm_state_path: str = ''
    default_prompt: str | None = None
    adapt_to_pi: bool = False
    local_files_only: bool = True
    repack_transforms: tyro.conf.Suppress[_transforms.Group | None] = None
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    dataset_root: str = ''
    multi_rerobot: bool = False
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    def create(self, metadata_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
     
        
        if self.norm_state_path:
            norm_stats_path = pathlib.Path(
                self.norm_state_path
                ) 
            
        else:
            norm_stats_path = pathlib.Path('assets/pi0/oatmeal_3029_joint_abs/norm_stats.json')  # target3- joint
            
            
        
        norm_stats = _normalize.deserialize_json(norm_stats_path.read_text())

        repack_transforms = self.repack_transforms


        data_transforms = _transforms.Group(
            inputs=[s1_policy.S1Inputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[0])],
            outputs=[s1_policy.S1Outputs(adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[1])],
        )


        if self.norm_state_path:
            if self.use_so3[0]:
                delta_action_mask = _transforms.make_bool_mask(9, 9, -1, 9, -1,-3)  # torso 9, left 10 right 10
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActionsSO3(mask= delta_action_mask, structure=[9, 9, -1, 9, -1, -3])],
                    outputs=[_transforms.AbsoluteActionsSO3(mask= delta_action_mask, structure=[9, 9, -1, 9, -1, -3])],
                )

            else:
                delta_action_mask = _transforms.make_bool_mask(7, -1,-24)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )

        return DataConfig(
            repo_id=self.repo_id,
            dataset_root=self.dataset_root,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=self.model_transforms,
            local_files_only=self.local_files_only,
            multi_rerobot=self.multi_rerobot,
            use_quantile_norm=True,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotS1DoubleHandDataConfig(DataConfigFactory):
    use_so3: list[bool] = dataclasses.field(default_factory=lambda: [False, False])
    repo_id: str|list[str] = "lerobot_so3_data_30hz"
    norm_state_path: str = ''
    default_prompt: str | None = None
    adapt_to_pi: bool = False
    local_files_only: bool = True
    repack_transforms: tyro.conf.Suppress[_transforms.Group | None] = None
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    dataset_root: str = ''
    multi_rerobot: bool = False


    def create(self, metadata_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
     
        if self.norm_state_path:
            norm_stats_path = pathlib.Path(
                norm_state_path
                ) 
            
        else:
            norm_stats_path = pathlib.Path('assets/pi0/oatmeal_3029_joint_abs/norm_stats.json')  # target3- joint
            
            
        
        norm_stats = _normalize.deserialize_json(norm_stats_path.read_text())

        repack_transforms = self.repack_transforms


        data_transforms = _transforms.Group(
            inputs=[s1_policy.S1Inputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[0])],
            outputs=[s1_policy.S1Outputs(adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[1])],
        )


        if self.norm_state_path:
            if self.use_so3[0]:
                # delta_action_mask = _transforms.make_bool_mask(9, -1, 9, -1,-3)  # torso 9, left 10 right 10
                # data_transforms = data_transforms.push(
                #     inputs=[_transforms.DeltaActionsSO3(mask= delta_action_mask, structure=[9, -1, 9, -1, -3])],
                #     outputs=[_transforms.AbsoluteActionsSO3(mask= delta_action_mask, structure=[9, -1, 9, -1, -3])],
                # )
                raise ValueError("功能未完成")

            else:
                delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1,-16)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )

        return DataConfig(
            repo_id=self.repo_id,
            dataset_root=self.dataset_root,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=self.model_transforms,
            local_files_only=self.local_files_only,
            multi_rerobot=self.multi_rerobot,
        )


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True) 
class TrainConfig2: # 训练配置类
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    # checkpoint_base_dir: str = "./checkpoints"
    checkpoint_base_dir: str = "/kpfs-cognition/waikei/codes/openpi-uncle/checkpoints" 

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1
    
    new_repo_id: str = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


@dataclasses.dataclass(frozen=True) 
class TrainConfig: # 训练配置类
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "/kpfs-cognition/waikei/codes/openpi-uncle/checkpoints"
    # checkpoint_base_dir: str = "/kpfs-regular/baifu/openpi/checkpoint/openpi0" 

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 40*8
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 0
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1
    

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [

    TrainConfig(
        name="debug",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250909_Clean_up_the_trash_s1_05', '20250916_Sweep_trash_s1_26'],  # 123808 + 42898 = 166706              
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/debug/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=13100,                                                                                        # 166706 * 20 / 32 / 8 = 13023
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=13100, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32,
        num_workers=0,
        wandb_enabled=False,
        log_interval=1,
        save_interval=2000
    ),

    ### V3 baseline finetuning
    ## PI-0
    # finegrained
    TrainConfig(
        name="pi_0_microwave_711_finegrained_200_items",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['0805_Microwave_complete_process'],   
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/microwave/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_microwave_711_finegrained_200_items/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=13100,                                                                                        # 166706 * 20 / 32 / 8 = 13023
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=13100, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2000,
    ),      
    TrainConfig(
        name="pi_0_pen_holder_finegrained_200_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26'],                   #  80636                               + 169296 + 70149 = 320081   , '20250917_Insert_pen_s1_5', '20250919_Insert_pen_s1_5'
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=3150*5,                                                                                # 80636 * 10 / 32 / 8 = 3150
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=3150*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=3150*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=2000,
    ),
    TrainConfig(
        name="pi_0_pen_holder_finegrained_400_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26','20250919_Insert_pen_s1_5'],                   #  80636 + 70149 =  150785                 + 169296 + 70149 = 320081   , '20250917_Insert_pen_s1_5', 
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=5890*5,                                                                                # 150785 * 10 / 32 / 8 = 5890
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=5890*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=5890*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=2000,
    ),
    TrainConfig(
        name="pi_0_pen_holder_finegrained_800_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26','20250919_Insert_pen_s1_5', '20250917_Insert_pen_s1_5'],                   #  80636 + 70149 =  150785                 + 169296 + 70149 = 320081   , , 
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=12503*5,                                                                                # 320081 * 10 / 32 / 8 = 12503
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=12503*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=12503*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=2000,
    ),
    # coarse
    TrainConfig(
        name="pi_0_pen_holder_coarse_200_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26'],                   #  80636                               + 169296 + 70149 = 320081   , '20250917_Insert_pen_s1_5'(400), '20250919_Insert_pen_s1_5'(199)
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=3150*5,                                                                                # 80636 * 10 / 32 / 8 = 3150
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=3150*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=3150*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=2000,
    ),
    TrainConfig(
        name="pi_0_pen_holder_coarse_400_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26','20250919_Insert_pen_s1_5'],                   #  80636 + 70149 =  150785                             + 169296 + 70149 = 320081   , '20250917_Insert_pen_s1_5'(400)
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=5890*5,                                                                                # 150785 * 10 / 32 / 8 = 5890
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=5890*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=5890*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=5890,
        keep_period=5890,
    ),
    TrainConfig(
        name="pi_0_pen_holder_coarse_800_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26','20250919_Insert_pen_s1_5', '20250917_Insert_pen_s1_5'],                   #  80636 + 70149 =  150785                 + 169296 + 70149 = 320081   , , 
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=12503*5,                                                                                # 320081 * 10 / 32 / 8 = 12503
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=12503*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=12503*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=12503,
        keep_period=12503,
    ),
    ## PI-05
    # finegrained
    TrainConfig(
        name="pi_05_pen_holder_finegrained_200_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26'],                   #  80636                               + 169296 + 70149 = 320081   , '20250917_Insert_pen_s1_5', '20250919_Insert_pen_s1_5'
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=3150*5,                                                                                # 80636 * 10 / 32 / 8 = 3150
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=3150*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=3150*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=2000,
    ),

    TrainConfig(
        name="pi_05_microwave_finegrained",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['0728_Microwave_fixed_S8', '0728_Microwave_fixed_S8_1', '0729_Microwave_fixed_S8', '0804_Microwave_complete_process', 
            '0805_Microwave_complete_process', '0807_Microwave_whole_process_s8'],                              # 195392(220) + 81268(92) + 137581(162) + 175111(200) + 96405(119) + 132444(155) = 818201    
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/microwave/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_microwave_711_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=159800,                                                                                # 818201 * 50 / 32 / 8 = 159804.883
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=159800, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=159800 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=2000,
    ),
    # coarse
    TrainConfig(
        name="pi_05_pen_holder_coarse_200_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26'],                   #  80636                               + 169296 + 70149 = 320081   , '20250917_Insert_pen_s1_5', '20250919_Insert_pen_s1_5'
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=3150*5,                                                                                # 80636 * 10 / 32 / 8 = 3150
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=3150*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=3150*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=2000,
    ),
    TrainConfig(
        name="pi_05_pen_holder_coarse_400_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26','20250919_Insert_pen_s1_5'],                   #  80636 + 70149 =  150785                             + 169296 + 70149 = 320081   , '20250917_Insert_pen_s1_5'(400)
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=5890*5,                                                                                # 150785 * 10 / 32 / 8 = 5890
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=5890*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=5890*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=5890,
        keep_period=5890,
    ),
    TrainConfig(
        name="pi_05_pen_holder_coarse_800_items_long_schedule",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26','20250919_Insert_pen_s1_5', '20250917_Insert_pen_s1_5'],                   #  80636 + 70149 =  150785                 + 169296 + 70149 = 320081   , , 
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_finegrained/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=12503*5,                                                                                # 320081 * 10 / 32 / 8 = 12503
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=12503*5, peak_lr=5e-5, decay_lr=5e-6, warmup_steps=12503*5 // 50),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=50,
        save_interval=12503,
        keep_period=12503, # This was already correct, but ensuring it's understood.
    ),













    ### Test Fixed 19th robot's data
    TrainConfig(
        name="pi_0_heat_cupcakes_w_fixed_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['0829_microwave_s1_19', '0901_microwave_s1_19', '20250915_Microwave_heating_s1_10',
                '20250916_Microwave_heating_s1_10', '20250918_place_in_cake_into_microwave_s1_19',
                '20250919_place_in_cake_into_microwave_s1_19', '20250923_place_in_cake_into_microwave_s1_19'], # 194376 + 131424 + 197165 + 80074 + 125311 + 130742 + 31755 = 890847
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_partial/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_heat_cupcakes_w_fixed_19/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            data_transforms = _transforms.Group(
                            inputs=[s1_policy.S1Inputs(action_dim=32, adapt_to_pi=False, use_so3=True)],
                            outputs=[s1_policy.S1Outputs(adapt_to_pi=False, use_so3=True)],
                        ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=69600,                                                                                # 890847 * 20 / 32 / 8 = 69597.42
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=69600, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=15000
    ),

    ### S1 diversity config
    #
    # PI-0 Configs
    #
    TrainConfig(
        name="pi_0_heat_cupcakes_wo_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Microwave_heating_s1_10', '20250916_Microwave_heating_s1_10'],                      # 197165 + 80074 = 277239
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_partial/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_heat_cupcakes_wo_19/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=21700,                                                                                # 277239 * 20 / 32 / 8 = 21659.2969 >> 21700
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=21700, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2000
    ),
    TrainConfig(
        name="pi_0_change_tissue_wo_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250916_change_tissues_s1_5', '20250916_Change_tissues_s1_26','20250917_change_tissues_s1_5', 
            '20250919_change_tissues_s1_5'],                                                                            # num_frames: 155775 + 141153 + 174605 + 66496 = 538029
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_partial/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_change_tissue_wo_19/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=42100,                                                                                                        # 538029 * 20 / 32 / 8 = 42033
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=42100, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2000
    ),
    TrainConfig(
        name="pi_0_pen_holder_wo_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26', '20250917_Insert_pen_s1_5', '20250919_Insert_pen_s1_5'],                   #  80636 + 169296 + 70149 = 320081
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_partial/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_wo_19/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=25_100,                                                                                # 320081 * 20 / 32 / 8 = 25006
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=25_100, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2000
    ),
    TrainConfig(
        name="pi_0_pnp_basketball_big_wo_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250919_pnp_basketball_big_s1_10'],                                                                          # 35022
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_partial/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pnp_basketball_big/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            state_drop_prob = 0.0,
            state_noise_prob = 0.0,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=2_200,                                                                                        # 35022 * 20 / 320 = 2188.875  |  35022 * 20 / 32 / 8 = 2736.09375
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=2_200, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=200
    ),    
    TrainConfig(
        name="pi_0_pack_gift_wo_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250922_pack_gifts_s1_10', '20250922_pack_toy_s1_10', '20250923_pack_gifts_s1_10',
            '20250926_pack_toy_s1_26', '20250926_pack_toy_yao_s1_26', '20250928_pack_toy_s1_5'],        #                                                                       # 35022
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pack_gift_wo_19/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            state_drop_prob = 0.0,
            state_noise_prob = 0.0,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=2_200,                                                                                        # 
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=2_200, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=40*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2200
    ),      
    TrainConfig(
        name="pi_0_sweep_table_and_pour_robot_5_26",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250909_Clean_up_the_trash_s1_05', '20250916_Sweep_trash_s1_26'],  # 123808 + 42898 = 166706              
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_sweep_table_and_pour_robot_5_26/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(
            action_horizon=32,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/kpfs-regular/baifu/openpi/openpi-assets/checkpoints/pi0_base/params"
                    ), 
        num_train_steps=13100,                                                                                        # 166706 * 20 / 32 / 8 = 13023
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=13100, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2000
    ),          
    # 
    # PI-0.5 Configs
    #
    TrainConfig(
        name="pi_05_heat_cupcakes_wo_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Microwave_heating_s1_10', '20250916_Microwave_heating_s1_10'],                      # 197165 + 80074 = 277239
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_partial/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_heat_cupcakes_wo_19/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=21700,                                                                                # 277239 * 20 / 32 / 8 = 21659.2969 >> 21700
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=21700, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2000
    ),
    TrainConfig(
        name="pi_05_change_tissue_wo_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250916_change_tissues_s1_5', '20250916_Change_tissues_s1_26','20250917_change_tissues_s1_5', 
            '20250919_change_tissues_s1_5'],                                                                            # num_frames: 155775 + 141153 + 174605 + 66496 = 538029
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_partial/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_change_tissue_wo_19/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=42100,                                                                                                        # 538029 * 20 / 32 / 8 = 42033
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=42100, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2000
    ),
    TrainConfig(
        name="pi_05_pen_holder_wo_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250915_Grab_pen_holder_s1_26', '20250917_Insert_pen_s1_5', '20250919_Insert_pen_s1_5'],                   #  80636 + 169296 + 70149 = 320081
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_partial/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pen_holder_wo_19/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=25_100,                                                                                # 320081 * 20 / 32 / 8 = 25006
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=25_100, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2000
    ),
    TrainConfig(
        name="pi_05_pnp_basketball_big_wo_19",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250919_pnp_basketball_big_s1_10'],                                                                          # 35022
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_partial/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_pnp_basketball_big/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=2_800,                                                                                        # 35022 * 20 / 32 / 8 = 2736.09375
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=2_800, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=200
    ), 
    TrainConfig(
        name="pi_05_sweep_table_and_pour_robot_5_26",            
        data=LeRobotS1DataConfig(
            use_so3 = [True,True],
            multi_rerobot=True,
            repo_id = ['20250909_Clean_up_the_trash_s1_05', '20250916_Sweep_trash_s1_26'],  # 123808 + 42898 = 166706              
            dataset_root = "/kpfs-regular/share_space/data/lerobot_data_aliyun/s1_data/diversity_finetune/",
            norm_state_path='/kpfs-cognition/waikei/codes/openpi-uncle/assets/pi_0_sweep_table_and_pour_robot_5_26/norm_stats.json',
            adapt_to_pi=False,
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "images_dict.head.rgb",
                                "cam_left_wrist": "images_dict.left.rgb",
                                "cam_right_wrist": "images_dict.right.rgb",
                            },
                            "state": "cartesian_so3_dict.cartesian_pose_state", 
                            "actions": "cartesian_so3_dict.cartesian_pose_command",
                            "prompt": "finegrained_prompt",
                            "sub_task_index":"sub_task_index",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]   # 0~28 前29维
                    )  
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi05_Config(
            action_horizon=32,
            pi05=True,
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    "/ks3-regular/waikei/ckpts/pi05_base/params"
                    ), 
        num_train_steps=13100,                                                                                        # 166706 * 20 / 32 / 8 = 13023
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=13100, peak_lr=5e-5, decay_lr=5e-6),
        batch_size=32*8,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        save_interval=2000
    ),      

  
]
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]


if __name__ == "__main__":
    import openpi.training.config as _config
    import sys
    sys.argv = [
        "train.py",   # 通常第一个参数随便写，代表脚本名
        "pi_0_diversity_partial_1",
        "--exp-name=pi_0_diversity_partial_1",
        # "--overwrite"
    ]
    print('debug')
    config = _config.cli()    
    print(config)