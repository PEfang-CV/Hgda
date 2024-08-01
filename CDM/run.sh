# Generate defect images by CDM 
MODEL_PATH="--checkmodel_path /root/data/gd/CDM/models/checkpoint.pt"
SAVE_PATH="--out_file /root/data/gd/CDM/Results/NEU-Seg-demo-Results-ddim50"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python sample.py $MODEL_PATH $SAVE_PATH $MODEL_FLAGS --classifier_scale 10.0  $SAMPLE_FLAGS --timestep_respacing ddim50
