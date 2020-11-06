python train.py \
--baseroot '/mnt/lustre/zhaoyuzhi/dataset/ILSVRC2012_train_256' \
--finetune_path './models/scribble_colorization_epoch10_batchsize16.pth' \
--multi_gpu True \
--checkpoint_interval 5 \
--finetune_path '' \
--multi_gpu True \
--epochs 41 \
--batch_size 4 \
--lr_g 1e-4 \
--lambda_l1 1 \
--lambda_perceptual 1 \
--lambda_gan 0.1 \
--lr_decrease_epoch 10 \
--lr_decrease_factor 0.5 \
--num_workers 8 \
--imgsize 256 \
--color_point 30 \
--color_width 5 \
--color_blur_width 11 \