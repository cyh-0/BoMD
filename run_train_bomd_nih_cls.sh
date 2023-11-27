CUDA_VISIBLE_DEVICES=0 \
python main_cls.py \
--load_mid_features \
--mid_ckpt "run-20220816_175557-6dn2k0id" \
--epochs_cls 40 \
--batch_size 16 \
--num_classes 14 \
--lr_cls 0.05 \
--train_data NIH \
--lam 0.6 \
--beta 0.5 \
--nsd_topk 10 \
--nsd_drop_th 1 \
--run_note "" \
--run_name "[BoMD-CLS]" \
--enhance_dist \
--tags BoMD-CLS \
--wandb_mode disabled
