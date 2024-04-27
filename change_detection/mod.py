import torch
from timm import utils
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from utils.parser import get_parser_with_args

parser, metadata = get_parser_with_args()
opt = parser.parse_args()
opt.distributed = True
device = utils.init_distributed_device(opt)


model = torch.load("outputs/change_detection/cdd_mixformer/exp1/checkpoint_epoch_199.pth",map_location="cpu")

torch.save(model.state_dict(),"checkpoint_epoch_199.pth")