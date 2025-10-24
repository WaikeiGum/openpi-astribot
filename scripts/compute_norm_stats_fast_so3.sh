
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export CUDA_VISIBLE_DEVICES=0

# uv run scripts/compute_norm_stats_fast_so3.py --config-name 0613_pi_tf_norm
# uv run scripts/compute_norm_stats_fast_so3.py --config-name 0613_togo_box_combine
# uv run scripts/compute_norm_stats_fast_so3.py --config-name 0613_towel_folding_combine

# uv run scripts/compute_norm_stats_fast_so3.py --config-name 0616_s1_so3
# uv run scripts/compute_norm_stats_fast_so3.py --config-name 0618_pegboard_so3


# wandb 

uv run scripts/compute_norm_stats_fast_so3.py --config-name pi_0_heat_cupcakes_wo_19

