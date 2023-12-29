python 2_augment.py --dataroot image_input --name HyperTest-resnet_6blocks-2-16-10-0.5 \
       --model test --netG resnet_6blocks  --ngf 16   --no_dropout --input_nc 1 \
       --output_nc 1  --checkpoints_dir trained_models \
       --results_dir image_output
