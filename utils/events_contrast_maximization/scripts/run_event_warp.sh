

python /mnt/workspace/ywb/WorldNew/evggt/utils/events_contrast_maximization/utils/events_cmax_fcr.py \
                    /mnt/workspace/ywb/WorldNew/evggt/utils/outputs/event.h5 \
                    --img_size 288 512  --save_dir /mnt/workspace/ywb/WorldNew/evggt/utils/outputs \
                    --save_vis \
                    --visualize_warped

python /mnt/workspace/ywb/WorldNew/evggt/utils/events_contrast_maximization/utils/events_cmax_fcr.py \
                    /mnt/workspace/ywb/WorldNew/evggt/data/interlaken_00_c_events_left/events.h5 \
                    --img_size 480 640  --save_dir /mnt/workspace/ywb/WorldNew/evggt/utils/outputs/interlaken_00_c_events_left \
                    --start_idx 20000 --num_events 100000
                    --save_vis \
                    --visualize_warped
python visualize_events.py /mnt/workspace/ywb/WorldNew/evggt/data/event.h5 
python visualize.py /mnt/workspace/ywb/WorldNew/evggt/data/event.h5 

python visualize_event_mp4.py /mnt/workspace/ywb/WorldNew/evggt/data/event.h5  --output /mnt/workspace/ywb/WorldNew/evggt/utils/outputs/event_video.mp4 --width 512 --height 288  --fps 60 --window 0.0005 --max_events 100000