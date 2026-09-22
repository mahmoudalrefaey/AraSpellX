from argparse import ArgumentParser
import torch


def validate_training_args(args):
    if args.n_gpus != 1:
        raise ValueError(
            'This project is configured for single-GPU CUDA training only. '
            'Please use --n_gpus 1.'
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA training is required, but CUDA is not available on this machine.'
        )
    if args.dist_key is None:
        ratio_key = f'distorted_{args.distortion_ratio}'
        args.dist_key = ratio_key
    return args


def add_model_args(parser) -> None:
    group = parser.add_argument_group('Model Args')
    group.add_argument(
        '--d_model', default=512, type=int,
        help='the model dimensionality'
    )
    group.add_argument(
        '--h', default=8, type=int,
        help='The number of heads'
        )
    group.add_argument(
        '--n_layers', default=4, type=int,
        help='The number of encoder and decoder layers'
        )
    group.add_argument(
        '--p_dropout', default=0.1, type=float,
        help='The dropout ratio'
    )
    group.add_argument(
        '--hidden_size', default=256, type=int,
        help='The model hidden dim'
    )
    group.add_argument(
        '--rnn_emb_size', default=512, type=int,
        help='The embedding size of the RNN model'
    )
    group.add_argument(
        '--bidirectional', default=False, action='store_true'
    )
    group.add_argument(
        '--model', default='transformer', type=str,
        help='The model architecture, transformer'
    )


def add_training_args(parser) -> None:
    group = parser.add_argument_group('Training Args')
    group.add_argument(
        '--epochs', default=3, type=int,
        help='The number of training epochs'
    )
    group.add_argument(
        '--batch_size', default=128, type=int,
        help='The training batch size'
    )
    group.add_argument(
        '--train_path', default='data/dataset/train.csv', type=str,
        help='The training data file path'
    )
    group.add_argument(
        '--clean_key', default='clean', type=str,
        help='The csv column name of the clean items'
    )
    group.add_argument(
        '--dist_key', default=None, type=str,
        help='The csv column name of the distorted items. If omitted, it is inferred from --distortion_ratio (for example: distorted_0.1).'
    )
    group.add_argument(
        '--test_path', default='data/dataset/test.csv', type=str,
        help='The testing data file path'
    )
    group.add_argument(
        '--max_len', default=128, type=int,
        help='The maximum length of each example'
    )
    group.add_argument(
        '--opt_betas', default=[0.9, 0.98], nargs='+',
        help='Adam Optimizer\'s beta0 and beta1'
    )
    group.add_argument(
        '--warmup_staps', default=4000, type=int,
        help='Adam Optimizer\'s warmup steps'
    )
    group.add_argument(
        '--opt_eps', default=1e-9, type=float,
        help='Adam Optimizer\'s eps, used to avoid dividing by zero'
    )
    group.add_argument(
        '--dist_backend', default='nccl', type=str,
        help='The distributed training backend'
    )
    group.add_argument(
        '--dist_port', default=12345, type=int,
        help='The distributed training port'
    )
    group.add_argument(
        '--n_gpus', default=1, type=int,
        help='The number of GPUs to train on. This project supports CUDA single-GPU training only.'
    )
    group.add_argument(
        '--pre_trained_path', default=None, type=str,
        help='The pretrained model path'
    )
    group.add_argument(
        '--outdir', default='outdir/', type=str,
        help='The path to save the checkpoints and the logs to'
    )
    group.add_argument(
        '--tokenizer_path', default='outdir/tokenizer.json', type=str,
        help='The path of the saved tokenizer, or to save the tokenizer to'
    )
    group.add_argument(
        '--alpha', default=0.1, type=float,
        help='Label smoothing value '
    )
    group.add_argument(
        '--stop_after', default=5, type=int,
        help='The number of epochs to stop after if no improvements happened'
    )
    group.add_argument(
        '--distortion_ratio', default=0.1, type=float,
        help='The data distortion/corruption ratio'
    )
    group.add_argument(
        '--logger_type', default='tensor_board', type=str,
        help='The logger type, either tensor_board or basic'
    )
    group.add_argument(
        '--logdir', default='outdir/logs', type=str,
        help='The directory to save the logs to'
    )
    group.add_argument(
        '--optim', default='adamw', type=str,
        help='The optimizer to use, either adam, adamw, or adamexp'
    )
    group.add_argument(
        '--lr', default=0.001, type=float,
        help='The learning rate, it is only used when adam optimizer used'
    )
    group.add_argument(
        '--decay_rate', default=0.001, type=float,
        help='The learning decay rate, it is only used when adamexp optimizer used'
    )
    group.add_argument(
        '--grad_norm', default=0.003, type=float,
        help='The maximum norm for gradiant clipping.'
    )
    group.add_argument(
        '--clip_grad', default=True, action='store_true'
    )
    group.add_argument(
        '--mixed_precision', default=True , action='store_true',
        help='Enable mixed precision training (BF16 on supported GPUs, FP16 otherwise)'
    )
    group.add_argument(
        '--num_workers', default=0, type=int,
        help='Number of data loader workers'
    )
    group.add_argument(
        '--pin_memory', default=True, action='store_true',
        help='Use pinned memory for data loading'
    )
    group.add_argument(
        '--log_interval', default=50, type=int,
        help='Log interval (steps) for TensorBoard'
    )
    group.add_argument(
        '--val_interval', default=1, type=int,
        help='Validation interval (epochs)'
    )
    group.add_argument(
        '--save_attention_viz', default=False, action='store_true',
        help='Save attention visualizations during validation'
    )
    group.add_argument(
        '--grad_accum_steps', default=2, type=int,
        help='Gradient accumulation steps (effective batch size = batch_size * grad_accum_steps)'
    )


def get_preprocessing_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--sep', default=[
            '\n', '\t', '.', '،', ',', '=', ':', '-', '\\', '/'
            ], nargs='+', type=str,
        help='The seperator to be used to split the lines on'
        )
    parser.add_argument(
        '--min_len', default=15, type=int,
        help='The minimum line length to keep'
        )
    parser.add_argument(
        '--max_len', default=128, type=int,
        help='The maximum line length to keep'
        )
    parser.add_argument(
        '--dist_run', default=False, action='store_true'
    )
    parser.add_argument(
        '--data_path', default='data/', type=str
    )
    parser.add_argument(
        '--save_path', default='clean_data.txt', type=str
    )
    parser.add_argument(
        '--max_rep_chars', default=2, type=str
    )
    parser.add_argument(
        '--execlude_words_files', default='words.json', type=str
    )
    parser.add_argument(
        '--max_oov', default=1, type=int
    )
    parser.add_argument(
        '--min_words', default=3, type=int
    )
    parser.add_argument(
        '--max_words', default=20, type=int
    )
    parser.add_argument(
        '--dist_ratios', default=[0.05, 0.1, 0.15], nargs='+'
    )
    return parser.parse_args()


def get_train_args():
    parser = ArgumentParser()
    add_model_args(parser)
    add_training_args(parser)
    args = parser.parse_args()
    return validate_training_args(args)


def get_transformer_args(
        args, voc_size: int, rank: int, pad_idx: int
        ) -> dict:

    enc_params = {
        'n_layers': args.n_layers,
        'voc_size': voc_size,
        'hidden_size': args.hidden_size,
        'p_dropout': args.p_dropout,
        'pad_idx': pad_idx
    }
    params = {
        'd_model': args.d_model,
        'h': args.h,
        'device': 'cuda:0',
        'voc_size': voc_size
    }
    dec_params = {
        'n_layers': args.n_layers,
        'p_dropout': args.p_dropout,
        'hidden_size': args.hidden_size,
        'voc_size': voc_size,
        'pad_idx': pad_idx
    }

    return {
        'enc_params': enc_params,
        'dec_params': dec_params,
        **params
    }


def get_model_args(args, voc_size, rank, pad_idx):
    if args.model == 'transformer':
        return get_transformer_args(args, voc_size, rank, pad_idx)
    raise ValueError(f"Unknown model: {args.model}")
