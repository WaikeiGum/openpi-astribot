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
    use_quantile_norm: bool = False

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
    use_delta_joint_actions: bool = False
    default_prompt: str | None = None
    adapt_to_pi: bool = False
    local_files_only: bool = True
    repack_transforms: tyro.conf.Suppress[_transforms.Group | None] = None
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    dataset_root: str = ''
    multi_rerobot: bool = False
    new_repo_id: str = None


    def create(self, metadata_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # norm_stats_path = pathlib.Path("/home/extra/baifu/openpi/pi0_base/assets/franka/norm_stats.json")
        # norm_stats_path = pathlib.Path('/home/lrq/Projects/openpi_ori/openpi/assets/pi0/id1/norm_stats_rightarm_padding.json')
        # norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/s1/id1/norm_stats_rightarm_padding.json')
        # norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_3014/norm_stats.json')  # target1
        # norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_target2_detla_joint/norm_stats.json')  # target2 delta joint
        # norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_3029_joint_abs/norm_stats.json')  # target3- joint
        
        if self.use_delta_joint_actions:
            norm_stats_path = pathlib.Path('assets/pi0/oatmeal_target2_detla_joint/norm_stats.json')  # target2 delta joint
            # norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_task9_dalta/norm_stats.json')  # target9 delta joint
            
        else:
            norm_stats_path = pathlib.Path('assets/pi0/oatmeal_3029_joint_abs/norm_stats.json')  # target3- joint
            
            
        
        norm_stats = _normalize.deserialize_json(norm_stats_path.read_text())

        repack_transforms = self.repack_transforms


        data_transforms = _transforms.Group(
            inputs=[s1_policy.S1Inputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[0])],
            outputs=[s1_policy.S1Outputs(adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[1])],
        )


        if self.use_delta_joint_actions:
            if self.use_so3[0]:
                delta_action_mask = _transforms.make_bool_mask(9, -1, 9, -1)
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
            new_repo_id = self.new_repo_id
        )


@dataclasses.dataclass(frozen=True)
class LeRobotS1DataConfig_popcorn(DataConfigFactory):
    use_so3: list[bool] = dataclasses.field(default_factory=lambda: [False, False])
    repo_id: str|list[str] = "lerobot_so3_data_30hz"
    use_delta_joint_actions: bool = False
    default_prompt: str | None = None
    adapt_to_pi: bool = False
    local_files_only: bool = True
    repack_transforms: tyro.conf.Suppress[_transforms.Group | None] = None
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    dataset_root: str = ''

    multi_rerobot: bool = False


    def create(self, metadata_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # norm_stats_path = pathlib.Path("/home/extra/baifu/openpi/pi0_base/assets/franka/norm_stats.json")
        # norm_stats_path = pathlib.Path('/home/lrq/Projects/openpi_ori/openpi/assets/pi0/id1/norm_stats_rightarm_padding.json')
        # norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/s1/id1/norm_stats_rightarm_padding.json')
        # norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_3014/norm_stats.json')  # target1
        # norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_target2_detla_joint/norm_stats.json')  # target2 delta joint
        # norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_3029_joint_abs/norm_stats.json')  # target3- joint
        norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_task9_dalta/norm_stats.json')  # target9- joint
        
        
        # if self.use_delta_joint_actions:
        #     norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_target2_detla_joint/norm_stats.json')  # target2 delta joint
        # else:
        #     norm_stats_path = pathlib.Path('/home/bai/project/openpi/assets/pi0/oatmeal_3029_joint_abs/norm_stats.json')  # target3- joint
        
        norm_stats = _normalize.deserialize_json(norm_stats_path.read_text())

        repack_transforms = self.repack_transforms


        data_transforms = _transforms.Group(
            inputs=[s1_policy.S1Inputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[0])],
            outputs=[s1_policy.S1Outputs(adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[1])],
        )


        if self.use_delta_joint_actions:
            if self.use_so3[0]:
                delta_action_mask = _transforms.make_bool_mask(9, -1, 9, -1)
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
    checkpoint_base_dir: str = "/cognition/baifu_ckpt/openpi0" 

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
    # checkpoint_base_dir: str = "./checkpoints"
    checkpoint_base_dir: str = "/cognition/baifu_ckpt/openpi0" 

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


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    TrainConfig(
        name="pi0_libero",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    # This config is used to demonstrate how to train on a simple simulated environment.
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),

  
    TrainConfig(
        name="pi0_s1_right_arm_oatmeal_task9_delta_split",
        # new_repo_id = [
        #                 "0428_flip", # 水平翻转
        #                "0428_1_turn_flip","0430_turn_flip", # 这个是转铲柄+ 水平翻转的
        #             # 
        #                 "0506_2","0508_4","0427_3_simple","0506_3","0428_2_blackseeds","0430_1_pick_again","0506_1","0429_2_pick_again",
        #                "0427_1_blackseeds","0430_2","0427_2_simple","0508_3","0429_1_pick_again","0507_1","0508_1","0508_2",
        #             # 9
        #                 "0509_2",
        #                "0513_1",
        #                "0513_2",
        #                "0514_1",
        #                "0514_2",
        #             #  10
        #                "0515_1",
        #                "0515_2",
        #                "0515_3"
        # ],
        data=LeRobotS1DataConfig(
            use_so3 = [False,False],
            multi_rerobot=True,
            repo_id = [
                       '0319',
                       '0319_2','0320','0320_2','0321','0324','0324_1','0324_2','0325_1','0325_2','0325_3','0325_4',   # target 1
                       "0326","0401","0402","0403_1","0403_2","0407","0408",  # target 2
                       # "0416","0416_1",                                         # 旋转
                       # "0417_1","0417_2","0417_4", 
                       "0417_3","0418","0418_1","0418_2", "0418_3","0418_4","0419","0421",
                       "0421_1","0421_2","0421_3",
                       "0422_1","0422_2","0422_3",
                       "0422_4","0423_1","0423_2","0423_3","0423_4",
                       "0424_3_pick_scoop",
                       "0424_1_scoop_after_pour", "0424_2_scoop_after_pour",  
                       "0425_4_single_scoop", 
                       "0425_5_single_scoop",
                        "0425_1_turn","0425_2_turn","0425_3_turn", 
                       
                    #    "0428_flip", # 水平翻转
                       "0428_1_turn_flip","0430_turn_flip", # 这个是转铲柄+ 水平翻转的
                       # 8
                       "0506_2","0508_4","0427_3_simple","0506_3","0428_2_blackseeds","0430_1_pick_again","0506_1","0429_2_pick_again",
                       "0427_1_blackseeds","0430_2","0427_2_simple","0508_3","0429_1_pick_again","0507_1","0508_1","0508_2",
                    #    # 9
                       "0509_2",
                       "0513_1",
                       "0513_2",
                       "0514_1",
                       "0514_2",
                    #  10
                       "0515_1",
                       "0515_2",
                       "0515_3"
                       ],
            dataset_root = "/cognition/lerobot_Oatmeal/lerobot_split",
            use_delta_joint_actions=True,
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
                            "state": "joints_dict.joints_position_state",
                            "actions": "joints_dict.joints_position_command",  # 此时 action 已被修改（如果需要）
                            "prompt": "prompt",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[14,15,16,17,18,19,20,21], [14,15,16,17,18,19,20,21]]
                    )  # only right arm
                ]
            ),
            model_transforms =_transforms.Group(
                    inputs=[
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt( # 分词处理器
                            _tokenizer.PaligemmaTokenizer(), # 分词器
                            # _tokenizer.T5TokenEmbedding(
                            #     t5_embed_dir='/home/extra/liuruiqiang/openpi/lang_emb/pnp_0208', 
                            #     max_len=20),
                            t5=False,
                            default_prompt='pick up the object and place to the plate'
                        ),
                        # _transforms.DebugPrintTransform(),
                    ]
                ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(action_horizon=32),
        # weight_loader=weight_loaders.GoogleViTWeightLoader(),
        # weight_loader=weight_loaders.CheckpointWeightLoader("/cognition/pi0_base/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #             "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task/basepi0oatmeal2/22000/params"),  # target 1
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #             "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task_delta/basepi0oatmeal_task1/58000/params"),  # target 2 泛化粗粒度
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #             "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task_delta_split/base_delta_task1/52000/params"),  # 泛化细粒度
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task2_delta_split/base_delta_task2/58000/params"   # target 2 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task3_delta_split/base_delta_task1/58000/params"  #   target 3 泛化细粒度
                    "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task5_delta_split/delta_task_5_v2/40000/params"  #   target 3 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task6_delta_split/delta_task_6/2000/params"     #   target 4 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task6_delta_split/delta_task_6/16000/params"    #   target 4 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task6_delta_split/delta_task_7/40000/params"     #   target 5 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task7_delta_split/delta_task_7/50000/params"  #   target 6 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task7_delta_split/delta_task_7/270000/params"  #   target 6 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task8_delta_split/delta_task_8/18000/params"
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task9_delta_split/delta_task_9/46000/params"  
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task9_delta_split/delta_task_10/60000/params"
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task9_delta_split/delta_task_11/4000/params"
                    # "/cognition/pi0_base/params"
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task9_delta_split/delta_task_14_only_turn/38000/params"  # 基于target 3 泛化细粒度

                    ), 
        num_train_steps=360_000,
        batch_size=16*7,
        num_workers=32,
        wandb_enabled=True,
        log_interval=100,
        # validate_interval=1000,
        save_interval=2000
    ),


    TrainConfig(
        new_repo_id = ['0319_2',"0325_4"],
        name="test",
        data=LeRobotS1DataConfig(
            use_so3 = [False,False],
            multi_rerobot=True,
            # repo_id = ['250314'],
            # repo_id = ['0319','0320','0320_2','0321','0324','250314'],
            repo_id = [
                       '0319',
                       '0319_2',
                        '0320','0320_2','0321','0324','0324_1','0324_2','0325_1','0325_2','0325_3','0325_4',   # target 1
                        #            "0326","0401","0402","0403_1","0403_2","0407","0408",  # target 2
                        #            # "0416","0416_1",                                         # 旋转
                        #            # "0417_1","0417_2","0417_4", 
                        #            "0417_3","0418","0418_1","0418_2", "0418_3","0418_4","0419","0421",
                        #            "0421_1","0421_2","0421_3",
                        #            "0422_1","0422_2","0422_3",
                        #            "0422_4","0423_1","0423_2","0423_3","0423_4",
                        #            "0424_3_pick_scoop",
                        #            "0424_1_scoop_after_pour", "0424_2_scoop_after_pour",  
                        #            "0425_4_single_scoop", 
                        #            "0425_5_single_scoop",
                        #            "0425_2_turn","0425_1_turn","0425_3_turn", 
                       ],
            dataset_root = "/mnt/nfs123/baifu/lerobot_split",
            use_delta_joint_actions=True,
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
                            "state": "joints_dict.joints_position_state",
                            "actions": "joints_dict.joints_position_command",  # 此时 action 已被修改（如果需要）
                            "prompt": "prompt",
                        }
                    ),
                    _transforms.GetDimRange(
                        key=['state', 'actions'],
                        dim=[[14,15,16,17,18,19,20,21], [14,15,16,17,18,19,20,21]]
                    )  # only right arm
                ]
            ),
            model_transforms =_transforms.Group(
                    inputs=[
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt( # 分词处理器
                            _tokenizer.PaligemmaTokenizer(), # 分词器
                            # _tokenizer.T5TokenEmbedding(
                            #     t5_embed_dir='/home/extra/liuruiqiang/openpi/lang_emb/pnp_0208', 
                            #     max_len=20),
                            t5=False,
                            default_prompt='pick up the object and place to the plate'
                        ),
                        # _transforms.DebugPrintTransform(),
                    ]
                ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=True,
        ),
        model=pi0.Pi0Config(action_horizon=32),
        # weight_loader=weight_loaders.GoogleViTWeightLoader(),
        # weight_loader=weight_loaders.CheckpointWeightLoader("/cognition/pi0_base/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #             "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task/basepi0oatmeal2/22000/params"),  # target 1
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #             "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task_delta/basepi0oatmeal_task1/58000/params"),  # target 2 泛化粗粒度
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #             "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task_delta_split/base_delta_task1/52000/params"),  # 泛化细粒度
        weight_loader=weight_loaders.CheckpointWeightLoader(
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task2_delta_split/base_delta_task2/58000/params"   # target 2 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task3_delta_split/base_delta_task1/58000/params"  #   target 3 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task5_delta_split/delta_task_5_v2/40000/params"  #   target 3 泛化细粒度
                    # "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task6_delta_split/delta_task_6/2000/params"
                    "/cognition/baifu_ckpt/openpi0/pi0_s1_right_arm_oatmeal_task6_delta_split/delta_task_6/16000/params"
                    ), 
        num_train_steps=360_000,
        batch_size=16*1,
        num_workers=48,
        wandb_enabled=True,
        log_interval=100,
        # validate_interval=1000,
        save_interval=2000
    ),
    

    

      
    TrainConfig(
        name="debug_single_batch",
        data=LeRobotS1DataConfig(
            use_so3 = [False,False],
            multi_rerobot=True,
            repo_id = ['250210_2'],  # 使用单个数据集
            dataset_root = "/cognition/lerobot_pnp/",
            use_delta_joint_actions=True,
            repack_transforms = _transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "images": {
                                    "cam_high": "images_dict.head.rgb",
                                    "cam_left_wrist": "images_dict.left.rgb",
                                    "cam_right_wrist": "images_dict.right.rgb",
                                },
                                "state": "joints_dict.joints_position_state",
                                "actions": "joints_dict.joints_position_command",  # 此时 action 已被修改（如果需要）
                                "prompt": "prompt",
                            }
                        ),
                        _transforms.GetDimRange(
                            key=['state', 'actions'],
                            dim=[[14,15,16,17,18,19,20,21], [14,15,16,17,18,19,20,21]]
                        )  # only right arm
                    ]
                ),
            model_transforms =_transforms.Group(
                        inputs=[
                            _transforms.ResizeImages(224, 224),
                            _transforms.TokenizePrompt(
                                _tokenizer.PaligemmaTokenizer(),
                                # _tokenizer.T5TokenEmbedding(
                                #     t5_embed_dir='/home/extra/liuruiqiang/openpi/lang_emb/pnp_0208', 
                                #     max_len=20),
                                t5=False,
                                default_prompt='pick up the object and place to the plate'
                            ),
                            _transforms.DebugPrintTransform(),
                        ]
                ),
            local_files_only=True,
        ),
        model=pi0.Pi0Config(action_horizon=32),
        weight_loader=weight_loaders.CheckpointWeightLoader("/cognition/pi0_base/params"),
        # ↓↓↓ 关键调试参数 ↓↓↓
        num_train_steps=5,        # 只跑5个step
        batch_size=1,             # 每个batch加载1条数据
        num_workers=1,            # 减少数据加载线程
        log_interval=1,          # 每个step都打印日志
        save_interval=10000,      # 关闭保存
        wandb_enabled=False       # 关闭wandb
    )
    
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
