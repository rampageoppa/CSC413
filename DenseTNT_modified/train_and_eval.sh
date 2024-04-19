epochs=12
batch=1 #64
lr=0.0005
dropout=0.5
output_dir="/home/guxunjia/project/DenseTNT_modified/test1"
train_dir=/home/guxunjia/project/HiVT_data/stream/train/data/
val_dir=/home/guxunjia/project/HiVT_data/stream/full_val/data/
python src/run.py --nuscenes --argoverse --argoverse2 --future_frame_num 30 --do_train --data_dir $train_dir --data_dir_for_val $val_dir --output_dir $output_dir --hidden_size 128 --train_batch_size $batch --use_map --core_num 16 --use_centerline --distributed_training 1 --other_params semantic_lane direction l1_loss goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph lane_scoring complete_traj complete_traj-3 --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1 --learning_rate $lr --hidden_dropout_prob $dropout --num_train_epochs $epochs
#for i in {7..12}
#do
i=7
echo $(python src/run.py --visualize --nuscenes --argoverse --argoverse2 --future_frame_num 30 --do_eval --data_dir $train_dir --data_dir_for_val $val_dir --output_dir $output_dir --hidden_size 128 --train_batch_size $batch --use_map --core_num 16 --use_centerline --distributed_training 1 --other_params semantic_lane direction l1_loss goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph lane_scoring complete_traj complete_traj-3 --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1 --learning_rate $lr --hidden_dropout_prob $dropout --model_recover_path $i ) >> $output_dir/eval_results
#done