CUDA_VISIBLE_DEVICES=0 \
python main_bomd.py \
--epochs_mid 30 \
--epochs_cls 40 \
--batch_size 64 \
--num_workers 16 \
--num_classes 14 \
--bert_name "bluebert" \
--embed_len 1024 \
--lr_mid 2e-4 \
--num_fea 3 \
--lr_cls 0.05 \
--wd_cls 0 \
--train_data "NIH" \
--lam 0.6 \
--beta 0.5 \
--relabel_method 3 \
--run_note "" \
--wandb_mode "disabled"