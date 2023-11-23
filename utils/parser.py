import argparse
import yaml
from timm import utils
# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below

def parser_init():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')


    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    # Keep this argument outside the dataset group because it is positional.
    parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                        help='path to dataset (positional is *deprecated*, use --data-dir)')
    parser.add_argument('--data-dir', metavar='DIR',
                        help='path to dataset (root dir)')
    parser.add_argument('--dataset', metavar='NAME', default='',
                        help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
    group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
    group.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
    group.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50")')
    group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
    group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
    group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
    group.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
    group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    group.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image size (default: None => model default)')
    group.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
    group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    group.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
    group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
    group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
    group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
    group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
    group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
    group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
    group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                    help='The number of steps to accumulate gradients (default: 1)')
    group.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')
    group.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
    group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
    group.add_argument('--head-init-scale', default=None, type=float,
                    help='Head initialization scale')
    group.add_argument('--head-init-bias', default=None, type=float,
                    help='Head initialization bias value')

    # scripting / codegen
    scripting_group = group.add_mutually_exclusive_group()
    scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                                help='torch.jit.script the full model')
    scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                                help="Enable compilation w/ specified backend (default: inductor).")

    # Optimizer parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd")')
    group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
    group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
    group.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
    group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
    group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
    group.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')
    group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

    # Learning rate schedule parameters
    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
    group.add_argument('--sched-on-updates', action='store_true', default=False,
                    help='Apply LR scheduler step on update instead of epoch end.')
    group.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate, overrides lr-base if set (default: None)')
    group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                    help='base learning rate: lr = lr_base * global_batch_size / base_size')
    group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                    help='base learning rate batch size (divisor, default: 256).')
    group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
    group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
    group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
    group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
    group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
    group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
    group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
    group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
    group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                    help='warmup learning rate (default: 1e-5)')
    group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
    group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
    group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
    group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                    help='epoch interval to decay LR')
    group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--warmup-prefix', action='store_true', default=False,
                    help='Exclude warmup period from decay schedule.'),
    group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10)')
    group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    group = parser.add_argument_group('Augmentation and regularization parameters')
    group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
    group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
    group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
    group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
    group.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
    group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
    group.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    group.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
    group.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    group.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    group.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
    group.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
    group.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
    group.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
    group.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
    group.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
    group.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
    group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
    group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
    group.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
    group.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
    group.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    group.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    group.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    group = parser.add_argument_group('Model exponential moving average parameters')
    group.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
    group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    group.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
    group.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
    group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
    group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
    group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
    group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
    group.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
    group.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    group.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
    group.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
    group.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
    group.add_argument('--synchronize-step', action='store_true', default=False,
                    help='torch.cuda.synchronize() end of each step')
    group.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
    group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
    group.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
    group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
    group.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    group.add_argument("--local_rank", default=0, type=int)
    group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
    group.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
    
    # modified
    group = parser.add_argument_group("modified settings")
    # lr and grad. update
    group.add_argument(
        "--auto-scale-warmup-min-lr",
        action="store_true",
        default=False,
        help="Auto-scale warmup and min lr with lr",
    )
    group.add_argument(
        "--use-deterministic",
        action="store_true",
        default=False,
        help="use deterministic settings",
    )
    group.add_argument(
        "--summary",
        default=None,
        type=str,
        metavar="SUMMARY",
        help="Lib for model summary.'torchinfo','torchstat','torchsummary','fvcore'. (default: None)",
    )
    group.add_argument(
        "--benchmark",
        default=None,
        type=str,
        metavar="BENCHMARK",
        help="Lib for model benchmark.'calflops','thop','ptflops'. (default: None)",
    )
    group.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Override experiment folder. (default: False)",
    )
    return config_parser, parser

def _parse_args():
    # Do we have a config file to parse?
    config_parser, parser = parser_init()
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

