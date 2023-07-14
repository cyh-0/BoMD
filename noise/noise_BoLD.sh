python -m noise.main_bomd_noise --epochs_val 20 \
--batch_size 64 --num_workers 16 --num_classes 14 \
--bert_name bluebert --embed_len 1024 --lr_pd 2e-4 \
--num_pd 3 --lr_cls 0.05 --wd_cls 0 --epochs_cls 30 \
--train_data NIH --lam 0.6 --beta 0.5 \
--run_note "test" --run_name "BoLD-BlueBert" \
--add_noise --noise_ratio 0.6 --noise_p 0.6 \
--wandb_mode online 


CUDA_VISIBLE_DEVICES=0 \
python -m noise.main_cls_noise \
--pd_ckpt "run-20221008_164557-1fu3000l" \
--epochs_cls 40 \
--batch_size 16 \
--num_classes 14 \
--lr_cls 0.05 \
--train_data "NIH" \
--lam 0.6 \
--beta 0.5 \
--run_note "" \
--run_name "BoLD-BlueBert" \
--enhance_dist \
--add_noise --noise_ratio 0.6 --noise_p 0.6 \
--tags "bluebert" \
--wandb_mode "online" \
