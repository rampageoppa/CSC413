epochs=12
batch=1 #64
lr=0.0002  #0.0005. 0.00035, 0.0001
wd=0.04  #0.01
dropout=0.2 # 0.5
output_dir="/home/guxunjia/project/DenseTNT_modified/final_models/maptr_al"
train_dir=/home/guxunjia/project/HiVT_data/maptr_bev/train/data/
val_dir=/home/guxunjia/project/HiVT_data/maptr_bev/full_val/data/

# output_dir="/home/guxunjia/project/DenseTNT_modified/stream_bev"
# train_dir=/home/guxunjia/project/HiVT_data/stream_bev/train/data/
# val_dir=/home/guxunjia/project/HiVT_data/stream_bev/full_val/data/
#   --temp_file_dir /home/guxunjia/project/DenseTNT_modified/test2/ \
CUDA_LAUNCH_BLOCKING=1
# python src/run.py \
#   --nuscenes \
#   --argoverse \
#   --argoverse2 \
#   --future_frame_num 30 \
#   --do_train \
#   --data_dir $train_dir \
#   --data_dir_for_val $val_dir \
#   --output_dir $output_dir \
#   --hidden_size 128 \
#   --train_batch_size $batch \
#   --use_map \
#   --core_num 16 \
#   --use_centerline \
#   --distributed_training 0 \
#   --other_params semantic_lane direction l1_loss goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph lane_scoring complete_traj complete_traj-3 \
#   --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1 \
#   --learning_rate $lr \
#   --weight_decay $wd \
#   --hidden_dropout_prob $dropout \
#   --num_train_epochs $epochs

# for i in {1..12}; do
i=6
echo $(python src/run.py \
  --visualize \
  --nuscenes \
  --argoverse \
  --argoverse2 \
  --future_frame_num 30 \
  --do_eval \
  --data_dir $train_dir \
  --data_dir_for_val $val_dir \
  --output_dir $output_dir \
  --hidden_size 128 \
  --train_batch_size $batch \
  --eval_batch_size 16 \
  --use_map \
  --core_num 16 \
  --use_centerline \
  --distributed_training 0 \
  --other_params semantic_lane direction l1_loss goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph lane_scoring complete_traj complete_traj-3 \
  --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1 \
  --learning_rate $lr \
  --weight_decay $wd \
  --hidden_dropout_prob $dropout \
  --model_recover_path $i) >> $output_dir/eval_results_$i
# done

