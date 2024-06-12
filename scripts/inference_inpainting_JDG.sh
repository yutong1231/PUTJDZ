
python /disks/sda/yutong2333/PUT-main/scripts/inference_inpainting.py  --func inference_inpainting \
--name  /disks/sda/yutong2333/PUT-main/OUTPUT/transformer_JDG_white_mask/checkpoint/000174e_617924iter.pth \
--input_res 256,256 \
--num_token_per_iter 2,5,10 --num_token_for_sampling 50,100 \
--image_dir /disks/sda/yutong2333/PUT-main/data/JDG/val \
--mask_dir /disks/sda/yutong2333/PUT-main/data/irregular-mask/test \
--save_masked_image \
--save_dir JDG \
--gpu 2