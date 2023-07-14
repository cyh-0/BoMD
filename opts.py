import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="CXR with MM")
    parser.add_argument("--root_dir", default="../dataset", type=str)
    parser.add_argument("--train_data", default="NIH", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr_pd", default=1e-4, type=float)
    parser.add_argument("--lr_cls", default=0.05, type=float)

    parser.add_argument("--wd_cls", default=0, type=float)
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_classes", default=14, type=int)

    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--total_runs", default=1, type=int)
    parser.add_argument("--save_dir", default="./ckpt", type=str)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--epochs_val", default=30, type=int)
    parser.add_argument("--epochs_cls", default=30, type=int)

    ##################      Optim      ##################

    ##################      Wandb      ##################
    parser.add_argument("--wandb_mode", default="disabled", type=str)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--run_note", default="", type=str)
    parser.add_argument("--tags", nargs="+", default="")

    ################## Hyper-parameter ##################
    parser.add_argument("--num_pd", default=3, type=int)
    parser.add_argument("--lam", default=0.6, type=float)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--smooth_epsilon", default=0.1, type=float)
    parser.add_argument(
        "--enhance_dist", default=True, action=argparse.BooleanOptionalAction
    )

    ##################      SGVAL      ##################
    parser.add_argument("--bert_name", default="bluebert", type=str)
    parser.add_argument("--embed_len", default=1024, type=int)
    parser.add_argument("--pd_ckpt", default="", type=str)
    parser.add_argument("--run_val", action="store_true")
    parser.add_argument("--load_val_features", action="store_true")
    parser.add_argument("--load_sample_graph", action="store_true")
    parser.add_argument("--relabel_method", default=3, type=int)
    parser.add_argument("--a2s_topk", default=10, type=int)
    parser.add_argument("--a2s_drop_th", default=1, type=int)

    # ##################       NPC       ##################
    # parser.add_argument("--npc_cls_ckpt", default=None, type=str)
    # parser.add_argument("--rho", default=5, type=float)
    # parser.add_argument("--lr_npc", default=0.001, type=float)

    ##################       Flag      ##################
    parser.add_argument("--use_ensemble", action="store_true")
    parser.add_argument("--load_pd_ckpt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_ckpt", default="", type=str)
    parser.add_argument(
        "--trim_data", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--ablation", default=False, action=argparse.BooleanOptionalAction
    )

    ##################       Noise      #################
    parser.add_argument(
        "--add_noise", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--noise_ratio", default=0.2, type=float)
    parser.add_argument("--noise_p", default=0.2, type=float)

    ##################       OLS      #################
    parser.add_argument("--ols", action="store_true")
    parser.add_argument("--elr_lambda", default=3.0, type=float)

    args = parser.parse_args()
    return args
