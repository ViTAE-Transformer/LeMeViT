import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from models.networks import BASE_Transformer
import calflops
# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

opt.distributed = False

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if opt.dataset == 'cdd':
    opt.dataset_dir = './datasets/CDD/'
elif opt.dataset == 'levir':
	opt.dataset_dir = './datasets/Levir/'

test_loader = get_test_loaders(opt)

#path = 'weights/sunet-32.pt'   # the path of the model
model = BASE_Transformer(opt, input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                            with_pos='learned', enc_depth=1, dec_depth=8).to(dev)

# print(calflops.calculate_flops(model,  args=[torch.randn(1 ,3, 256, 256),torch.randn(1,3,256,256)], output_as_string=True, output_precision=4))

pretrained_model = torch.load(opt.pretrained, map_location='cpu')
# print(pretrained_model.keys())
pretrained_model = {key.replace("module.", ""): value for key, value in pretrained_model.items()}
# print(pretrained_model.keys())
print(model.load_state_dict(pretrained_model))


c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels, fname in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)
        #cd_preds = cd_preds[-1] # BIT输出不是tuple
        _, cd_preds = torch.max(cd_preds, 1)
        
        #print(cd_preds.shape, labels.shape, fname)

        tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                        cd_preds.data.cpu().numpy().flatten(),labels=[0,1]).ravel()

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))
