# python eval.py --root /home/guxunjia/project/HiVT_data/ --batch_size 32 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/pt_rep/gt/checkpoints/epoch=61-step=28705.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data_w_uncertainty/ --batch_size 32 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/pt_rep/maptr_centerline/checkpoints/epoch=61-step=29449.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data_stream/ --batch_size 32 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/stream/checkpoints/epoch=58-step=27788.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data_maptrv2_cent/ --batch_size 32 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/pt_rep/maptr_centerline_std_trial2/checkpoints/epoch=60-step=28974.ckpt

# python eval.py --root /home/guxunjia/project/HiVT_data/HiVT_data_maptr/ --batch_size 32 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/maptr/checkpoints/epoch=60-step=28852.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data/HiVT_data_maptr/ --batch_size 32 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/maptr_std/checkpoints/epoch=62-step=29798.ckpt

##### Rebuttal Test #####
# python eval.py --root /home/guxunjia/project/HiVT_data/HiVT_data_maptr_cali/ --batch_size 32 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/maptr/checkpoints/epoch=60-step=28852.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data/HiVT_data_maptr_cali/ --batch_size 32 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/maptr_std/checkpoints/epoch=62-step=29798.ckpt


###### BEV Test ######
### MapTR BEV ###
# python eval.py --root /home/guxunjia/project/HiVT_data/maptr_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/maptr_al/checkpoints/epoch=63-step=120959.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data/maptr_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/maptr_al_unc/checkpoints/epoch=63-step=120959.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data/maptr_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/maptr_ab/checkpoints/epoch=54-step=103949.ckpt

### Stream BEV ###
# python eval.py --root /home/guxunjia/project/HiVT_data/stream_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/stream_al/checkpoints/epoch=63-step=120575.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data/stream_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/stream_al_unc/checkpoints/epoch=51-step=97967.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data/stream_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/stream_ab/checkpoints/epoch=48-step=92315.ckpt

### MapTRv2 BEV ###
# python eval.py --root /home/guxunjia/project/HiVT_data/maptrv2_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/maptrv2_al/checkpoints/epoch=63-step=120959.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data/maptrv2_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/maptrv2_al_unc/checkpoints/epoch=61-step=117179.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data/maptrv2_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/maptrv2_ab/checkpoints/epoch=60-step=115289.ckpt

### MapTRv2 Centerline BEV ###
# python eval.py --root /home/guxunjia/project/HiVT_data/maptrv2_cent_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/maptrv2_cent_al/checkpoints/epoch=63-step=121535.ckpt
# python eval.py --root /home/guxunjia/project/HiVT_data/maptrv2_cent_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/maptrv2_cent_al_unc/checkpoints/epoch=63-step=121535.ckpt
python eval.py --root /home/guxunjia/project/HiVT_data/maptrv2_cent_bev/ --batch_size 8 --ckpt_path /home/guxunjia/project/HiVT_modified/lightning_logs/bev/maptrv2_cent_ab/checkpoints/epoch=62-step=119636.ckpt
