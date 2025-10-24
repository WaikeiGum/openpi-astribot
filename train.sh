source ~/.bashrc
cd /kpfs-cognition/waikei/codes/openpi-uncle
source /kpfs-cognition/waikei/codes/openpi-uncle/.venv/bin/activate

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_0_diversity_partial_1 --exp-name=pi_0_diversity_partial_1 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_0_diversity_partial_1 --exp-name=pi_0_diversity_partial_1  --overwrite >> logs/pi0_diversity_partial.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_05_diversity_partial_1 --exp-name=pi_05_diversity_partial_1  --overwrite >> logs/pi_05_diversity_partial_1.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_0_make_oat_meal --exp-name=pi_0_make_oat_meal  --overwrite >> logs/pi_0_make_oat_meal.log



source ~/.bashrc
cd /kpfs-cognition/waikei/codes/openpi-uncle
source /kpfs-cognition/waikei/codes/openpi-uncle/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_05_heat_cupcakes --exp-name=pi_05_heat_cupcakes  --overwrite >> logs/pi_05_heat_cupcakes.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_0_heat_cupcakes_w_fixed_19 --exp-name=pi_0_heat_cupcakes_w_fixed_19  --overwrite >> logs/pi_0_heat_cupcakes_w_fixed_19.log

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_0_sweep_table_and_pour_robot_5_26 --exp-name=pi_0_sweep_table_and_pour_robot_5_26  --overwrite 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_05_pen_holder_finegrained_200_items_long_schedule --exp-name=pi_05_pen_holder_finegrained_200_items_long_schedule
