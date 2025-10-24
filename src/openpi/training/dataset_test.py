
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
class DataConfig:
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
        return dataclasses.replace(
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


    def create(self, metadata_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # norm_stats_path = pathlib.Path("/home/extra/baifu/openpi/pi0_base/assets/franka/norm_stats.json")
        norm_stats_path = pathlib.Path('/home/lrq/Projects/openpi_ori/openpi/assets/pi0/id1/norm_stats_rightarm_padding.json')
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
                delta_action_mask = _transforms.make_bool_mask(7, -1, -24)
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



data=LeRobotS1DataConfig(
    use_so3 = [False,False],
    multi_rerobot=True,
    repo_id = ['250210_2'],
    dataset_root = "/cognition/lerobot_pnp/",
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
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(),
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
)


print(data)


