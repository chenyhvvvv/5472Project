# Adult mouse brain
# First run the pipeline in notebooks/process_mouse_brain.ipynb to generate the data pickle file
CUDA_VISIBLE_DEVICES=3 python ./python_scripts/run_stitch3d.py --json_path ./configs/mouse_brain.json --run_name mouse_brain
CUDA_VISIBLE_DEVICES=1 python ./python_scripts/run_stitch3d.py --json_path ./configs/mouse_brain.json --run_name mouse_brain

# Human heart
CUDA_VISIBLE_DEVICES=1 python ./python_scripts/run_stitch3d.py --json_path ./configs/human_heart.json --run_name human_heart

# DLPFC
CUDA_VISIBLE_DEVICES=2 python ./python_scripts/run_stitch3d.py --json_path ./configs/dlpfc.json --run_name dlpfc