# LASC
# Datasets
# Running LASC
## source training
python3 main.py \
  --dataset <source_dataset> \
  --data_root <path_to_source_dataset> \
  --data_aug \
  --lr 0.1 \
  --crop_size 768 \
  --batch_size 2 \
  --freeze_BB \
  --ckpts_path saved_ckpts
## Feature optimization
python3 PIN_aug.py \
--dataset <source_dataset> \
--data_root <path_to_source_dataset> \
--total_it 100 \
--resize_feat \
--domain_desc <target_domain_description>  \
--save_dir <directory_for_saved_statistics>
