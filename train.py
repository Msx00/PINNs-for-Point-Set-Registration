import torch
import numpy as np
import argparse
import torch.utils.data
import datetime
import logging
import os
from tqdm import tqdm
from torch import nn
from loss import chamfer_loss, linear_elastic_loss, mae_loss
import random
from pathlib import Path
import matplotlib.pyplot as plt
import logbox

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def set_random_seed(seed=100):#0-几百万
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def options(argv=None):
    parser = argparse.ArgumentParser(description='Non-Rigid Registration')

    parser.add_argument('--seed', default="42", required=False, type=int, metavar='BASENAME', help='random seed')
    parser.add_argument('-o', '--log_dir', default=None, required=False, type=str, metavar='BASENAME', help='output filename (prefix)')

    parser.add_argument('--exp_name', type=str, default='log_message', metavar='N',
                            help='Name of the experiment')
    
    parser.add_argument('--traindata-path',
                        default="/home/data/msx/Project/MICCAI_PINNS/dataset/Simulation/resample/", 
                        required=False,
                        type=str,
                        metavar='PATH', help='path to the input training dataset')  # ..//表面点+内部点数据
    
    parser.add_argument('--traindata_yp_path',
                        default="/home/data/msx/Project/MICCAI_PINNS/dataset/Simulation/dataset/sampled_points_surface_all_patients/", 
                        required=False,
                        type=str,
                        metavar='PATH', help='path to the input training dataset')  # ..//表面点+内部点数据
        
    parser.add_argument('--YM',
                        default="/home/data/msx/Project/MICCAI_PINNS/dataset/Simulation/dataset/YM_PR/YMs.npy", 
                        required=False,
                        type=str,
                        metavar='PATH', help='path to the input training dataset')  # ..//表面点+内部点数据

    parser.add_argument('--PR',
                        default="/home/data/msx/Project/MICCAI_PINNS/dataset/Simulation/dataset/YM_PR/PRs.npy", 
                        required=False,
                        type=str,
                        metavar='PATH', help='path to the input training dataset')  # ..//表面点+内部点数据
        
    parser.add_argument('--useattention', default=False, type= bool, help='attention founction')
    
    parser.add_argument('--device', default='cuda:0', type=str, metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--dim-k', default=1024, type=int, metavar='K', help='dim. of the feature vector (default: 1024)')

    parser.add_argument('-b', '--batch-size', default=2, type=int, metavar='N', help='mini-batch size (default: 2)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], metavar='METHOD',
                        help='name of an optimizer (default: Adam)')
    
    parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

    parser.add_argument('--surfacechamfer', default='surfaceonly', type=str,help='use surface points or all points in the chamfer loss')
    
    parser.add_argument('--lr', default=1e-4, type=float, help='in the chamfer loss')

    # settings for training
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: null (no-use))')
    
    parser.add_argument('--checkpoint', default='', type=str, help='the tranning pth')
    parser.add_argument('--pretrained', default='./log/2024-12-27_17-02/checkpoints/checkpoints_snap_best.pth', type=str, help='the pre-trained pth')
    
    parser.add_argument('--use-Elasticloss', default=True, type=bool, help='use elastic loss')
    parser.add_argument('--usemaeloss', default=False, type=bool, help='use elastic loss')
    
    parser.add_argument('--elascale', default= 1e-5, type= float, help='use elastic loss')

    parser.add_argument('--active', default="LeakyReLU", type= str, help='active founction')

    args = parser.parse_args(argv)
    return args

    # if args.active == "Tanh":
    #     model = deform_four_by_four(torch.nn.Tanh(), args.useattention)
    # if args.active == "GELU":
    #     model = deform_four_by_four(torch.nn.GELU(), args.useattention)
    # if args.active == "Softplus":
    #     model = deform_four_by_four(torch.nn.Softplus(), args.useattention)
    # if args.active == "ReLU":
    #     model = deform_four_by_four(torch.nn.ReLU(), args.useattention)
    # if args.active == "ELU":
    #     model = deform_four_by_four(torch.nn.ELU(), args.useattention)
    # if args.active == "Sigmoid":
    #     model = deform_four_by_four(torch.nn.Sigmoid(), args.useattention)
    # if args.active == "LeakyReLU":
    #     model = deform_four_by_four(torch.nn.LeakyReLU(), args.useattention)  

def get_sample(data_dir):
    pathlist = []
    for i in range(0, 108):
        # path = data_dir + str(i) + "normalise_mr_us_1_0_4_1024_3.npy"
        path = data_dir + str(i) + ".npy"
        if os.path.exists(path):
            pathlist.append(path)
        else:
            print("没有路径", path)
    return pathlist



def load_us_centroid_distance(num): 
    abs_dir = "/home/data/msx/Project/Train/testdata"
    mr_us_centroidpath = abs_dir + "/label0/centroid/" + str(num) + "_mr_us_centroid.npy"
    mr_us_distancepath = abs_dir + "/label0/furthest distance/" + str(num) + "_mr_us_furthest_distance.npy"

    if os.path.exists(mr_us_centroidpath) and os.path.exists(mr_us_distancepath):
        centroid_data = np.load(mr_us_centroidpath)
        distance_data = np.load(mr_us_distancepath)
        mr_centroid = centroid_data[0]
        us_centroid = centroid_data[1]
        mr_furthest_distance = distance_data[0]
        us_furthest_distance = distance_data[1]
    return mr_centroid, us_centroid, mr_furthest_distance, us_furthest_distance


def flatten(x):
    return x.view(x.size(0), -1)


def mlp_layers(activatefounction, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(activatefounction)

        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp

    return layers


class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """

    def __init__(self, activatefounction, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(activatefounction, nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


def mlp_layers_wo_relu(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    '''
    It is the same with mlp_layers function, except that the ReLU layer is removed. 
    '''
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


class MLPNet_wo_relu(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """

    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers_wo_relu(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    return a


class TNet(torch.nn.Module):
    """ [B, K, N] -> [B, K, K]
    """
    def __init__(self, K):
        super().__init__()
        # [B, K, N] -> [B, K*K]
        self.mlp1 = torch.nn.Sequential(*mlp_layers(K, [64, 128, 1024], b_shared=True))
        self.mlp2 = torch.nn.Sequential(*mlp_layers(1024, [512, 256], b_shared=False))
        self.lin = torch.nn.Linear(256, K * K)

        for param in self.mlp1.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.mlp2.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.lin.parameters():
            torch.nn.init.constant_(param, 0.0)

    def forward(self, inp):
        K = inp.size(1)
        N = inp.size(2)
        eye = torch.eye(K).unsqueeze(0).to(inp)  # [1, K, K]

        x = self.mlp1(inp)
        x = flatten(torch.nn.functional.max_pool1d(x, N))
        x = self.mlp2(x)
        x = self.lin(x)

        x = x.view(-1, K, K)
        x = x + eye
        return x


class tranformer_offset_attention(nn.Module):
    def __init__(self, channels):
        super(tranformer_offset_attention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x = x + xyz
        # print("x1.shape",x.shape)

        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        # print("x2.shape",x.shape)

        return x
    
    
class PointNet_features(torch.nn.Module):  # //non rigid has no constraint on TNet// 非刚性没有约束TNET
    def __init__(self, activatefounction, useattention, num_c=3, dim_k=1024, use_tnet=False, sym_fn=symfn_max, scale=1):
        super().__init__()
        mlp_h1 = [int(64 / scale), int(64 / scale)]
        mlp_h2 = [int(64 / scale), int(128 / scale), int(dim_k / scale)]

        self.useattention = useattention

        self.h1 = MLPNet(activatefounction, num_c, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(activatefounction, mlp_h1[-1], mlp_h2, b_shared=True).layers
        if self.useattention:
            self.attention64 = tranformer_offset_attention(64)
            self.attention1024 = tranformer_offset_attention(dim_k)
        self.sy = sym_fn

        self.tnet1 = TNet(3) if use_tnet else None  # // bool
        self.tnet2 = TNet(mlp_h1[-1]) if use_tnet else None

        self.t_out_t2 = None
        self.t_out_h1 = None


    def forward(self, points):
        """ points -> features
            [B, N, 4] -> [B, K]
        """
        x = points.transpose(1, 2) 
        if self.tnet1:
            t1 = self.tnet1(x)  # // to check
            x = t1.bmm(x)

        x = self.h1(x)
        # print("x.shape",x.shape)
        if self.useattention:
            x = self.attention64(x)
        if self.tnet2:
            t2 = self.tnet2(x)
            self.t_out_t2 = t2
            x = t2.bmm(x)
        self.t_out_h1 = x  # local features

        x = self.h2(x)
        if self.useattention:
            x = self.attention1024(x)
        x = flatten(self.sy(x))
        return x  # //plot
   


class deform_four_by_four(torch.nn.Module):
    def __init__(self, activatefounction, useattention, num_c=3, dim_k=1024):
        super().__init__()
        self.ptfeatures = PointNet_features(activatefounction, useattention, num_c, dim_k, use_tnet=False)
        mlp_list_layers2 = [int(1024), int(512), int(256), int(128), int(64)]
        self.list_layers2 = MLPNet(activatefounction, 2 * dim_k + 3, mlp_list_layers2, b_shared=True).layers
        last_layer = [int(3)]
        self.last_layer = MLPNet_wo_relu(64, last_layer, b_shared=True).layers

        self.ptfeatures_pinns = PointNet_features(activatefounction, useattention, num_c, dim_k, use_tnet=False) #..// SIGMA
        self.pinns_strain1    = MLPNet(activatefounction, 2 * dim_k + 3, mlp_list_layers2, b_shared=True).layers
        self.pinns_strain2    = MLPNet_wo_relu(64, [int(6)], b_shared=True).layers  

    def forward(self, data):
        # data.requires_grad = True   #梯度断裂处
        source = data[:, 0, :, :]
        target = data[:, 1, :, :]
        pffeat_src = self.ptfeatures(source)
        pffeat_target = self.ptfeatures(target)
        global_feature = torch.cat((pffeat_src, pffeat_target), -1)
        num_source = source.shape[1]
        global_feature_repeated = global_feature.unsqueeze(1).repeat(1, num_source, 1)
        global_feature_repeated_conca = torch.cat((global_feature_repeated, source), -1)
        displacements_source_before_last_layer = self.list_layers2(global_feature_repeated_conca.permute(0, 2, 1))
        displacements_source = self.last_layer(displacements_source_before_last_layer)
        deformed_source = source + displacements_source.permute(0, 2, 1)

        pffeat_strain_source = self.ptfeatures_pinns(source)
        pffeat_strain_target = self.ptfeatures_pinns(target)
        global_feature_strain= torch.cat((pffeat_strain_source, pffeat_strain_target), -1)
        num_source = source.shape[1]
        global_feature_strain_repeated = global_feature_strain.unsqueeze(1).repeat(1, num_source, 1)
        global_feature_strain_repeated_conca = torch.cat((global_feature_strain_repeated, source), -1)
        strain_source_before_last_layer = self.pinns_strain1(global_feature_strain_repeated_conca.permute(0, 2, 1))
        strain_source = self.pinns_strain2(strain_source_before_last_layer)

        return deformed_source, displacements_source, strain_source
    


def downsample_mri_us(sample_return, number_of_samples=1024):
    # the input 1. is the list that contains the mri and us points
    #           2. the number of points in the sampled shape
    mri_points_original = sample_return[0]
    us_points_original = sample_return[1]

    indices_mri = np.random.choice(mri_points_original.shape[0], number_of_samples, replace=False)
    indices_us = np.random.choice(us_points_original.shape[0], number_of_samples, replace=False)

    mri_points_downsampled = mri_points_original[indices_mri, :]
    us_points_downsampled = us_points_original[indices_us, :]

    sample_return_downsampled = np.zeros([2, number_of_samples, 3])

    sample_return_downsampled[0, :, :] = mri_points_original
    sample_return_downsampled[1, :, :] = us_points_original
    return sample_return_downsampled


def convert_list_into_array(sample_return):
    # This script implements the part that converts the data list into array, but without downsampling
    #  Input is a list while the output is an array.
    number_of_samples = sample_return[0].shape[0]
    sample_return_downsampled = np.zeros([2, number_of_samples, 3])

    sample_return_downsampled[0, :, :] = sample_return[0]
    sample_return_downsampled[1, :, :] = sample_return[1]

    return sample_return_downsampled


class prostateset(torch.utils.data.Dataset):
    def __init__(self, rootdir, fileloader, transform=None, downsampleornot=False):
        samples = get_sample(rootdir)
        self.rootdir = rootdir
        self.fileloader = fileloader
        self.samples = samples
        self.transform = transform
        self.downsampleornot = downsampleornot  # whether we downsample the data again

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]

        sample = self.fileloader(path, allow_pickle=True)
        if self.transform is not None:
            sample = self.transform(sample)
        sample_return = []
        sample_return.append(sample[0])
        sample_return.append(sample[1])
        #        print(sample[0].shape)
        if self.downsampleornot:
            sample_return_downsampled = downsample_mri_us(sample_return)
        else:
            sample_return_downsampled = convert_list_into_array(sample_return)
        return sample_return_downsampled

class prostateset_v2(torch.utils.data.Dataset):
    '''
    This class implements that the data is read
    This definition is copied from 'non_rigid_registration_pinns_modified_v10.py'
    '''

    def __init__(self, ALL_SAMPLES, transform=None, downsampleornot=False, dataAugment=False):
        self.samples = ALL_SAMPLES
        self.transform = transform
        self.downsampleornot = downsampleornot  # whether we downsample the data again

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):  # 数据增强
        sample = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        sample_return = []
        sample_return.append(sample[0])
        sample_return.append(sample[1])
        if self.downsampleornot:
            sample_return_downsampled = downsample_mri_us(sample_return)
        else:
            sample_return_downsampled = convert_list_into_array(sample_return)
        return sample_return_downsampled


def save_checkpoint(state, filename, suffix):
    torch.save(state, '{}_{}.pth'.format(filename, suffix))



def train_1(model, trainloader, optimizer, device, surfacechamfer, args, log, YM_extracted_rows, PR_extracted_row):
    model.train()
    vloss = 0.0
    sum_cham_loss = 0.0
    sum_residual_strain, sum_equilibrium, sum_strain_energy, sum_mae = 0.0,0.0,0.0, 0.0
    count = 0
    # elas_loss_list = []
    for i, data in enumerate(trainloader):
        data = data.float().to(device)
        data.requires_grad = True 
        deformed, displacements, strain_source = model(data)

        if surfacechamfer == 'surfaceonly':
            num_points_used_in_loss_cham = 1024 #..// or 512
            loss_cham = chamfer_loss(deformed[:, 0:num_points_used_in_loss_cham, :],
                                     data[:, 1, 0:num_points_used_in_loss_cham, :],
                                     num_points_used_in_loss_cham)
            print('training loss_cham', loss_cham.item())
        else:
            loss_cham = chamfer_loss(deformed, data[:, 1, :, :], data.shape[2])
            print('training loss_cham', loss_cham.item())


        if args.use_Elasticloss:
            residual_strain, equilibrium, strain_energy = linear_elastic_loss(data, displacements, strain_source, YM_extracted_rows, PR_extracted_row)
        else:
            residual_strain, equilibrium, strain_energy = 0,0,0

        if args.usemaeloss:
            mae = mae_loss(deformed[:, :, :], data[:, 1, :, :])
            mae = float(mae)
        else:
            mae = 0


        #  -----------------compute total loss-----------------------
        w1 = 1e-3
        w2 = 1e-3
        w3 = 1e-5

        total_loss = loss_cham + mae + w1 * residual_strain + w2 * equilibrium + w3 * strain_energy

        outstr = 'total_loss %.14f,loss_cham %.14f,residual_strain %.14f, equilibrium: %.14f, strain_energy: %.14f' % (
        total_loss, loss_cham, residual_strain, equilibrium, strain_energy)

        log.cprint(outstr)

        # forward + backward + optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        vloss += total_loss.item()
        sum_cham_loss += loss_cham
        sum_residual_strain += residual_strain
        sum_equilibrium += equilibrium
        sum_strain_energy += strain_energy
        sum_mae += mae
        count += 1

    ave_vloss = float(vloss) / count
    ave_cham_loss = float(sum_cham_loss) / count
    ave_residual_strain = float(sum_residual_strain) / count
    ave_equilibrium = float(sum_equilibrium) / count
    ave_strain_energy = float(sum_strain_energy) / count
    ave_mae = float(sum_mae) / count

    return ave_vloss, ave_cham_loss, ave_residual_strain, ave_equilibrium, ave_strain_energy,ave_mae


def eval_1(model, testloader, device):
    model.eval() 
    vloss = 0
    count = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            data = data.float().to(device)
            deformed, displacements_source, strain_source= model(data)
            ''' use all points in the chamfer loss '''
            num_points_used_in_loss_cham = 1024
            loss_cham = chamfer_loss(deformed[:, 0:num_points_used_in_loss_cham, :],
                                     data[:, 1, 0:num_points_used_in_loss_cham, :],
                                     num_points_used_in_loss_cham)
            print('vali loss_cham', loss_cham.item())
            vloss += loss_cham.item()
            count += 1

    ave_vloss = float(vloss) / count
    return ave_vloss

def generate_dir(args):
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)

    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)    
    checkpoints_dir = checkpoints_dir / "checkpoints/"

    image_dir = exp_dir.joinpath('logimage/')
    image_dir.mkdir(exist_ok=True)

    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    log_trainningloss_dir = log_dir
    log_valiloss_dir = log_dir
    return checkpoints_dir, image_dir, log_trainningloss_dir, log_valiloss_dir

def run(args,trainset, YM_extracted_rows, PR_extracted_row):
    checkpoints_dir, image_dir, log_trainningloss_dir, log_valiloss_dir = generate_dir(args)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # create model
    if args.active == "Tanh":
        model = deform_four_by_four(torch.nn.Tanh(), args.useattention)
    if args.active == "GELU":
        model = deform_four_by_four(torch.nn.GELU(), args.useattention)
    if args.active == "Softplus":
        model = deform_four_by_four(torch.nn.Softplus(), args.useattention)
    if args.active == "ReLU":
        model = deform_four_by_four(torch.nn.ReLU(), args.useattention)
    if args.active == "ELU":
        model = deform_four_by_four(torch.nn.ELU(), args.useattention)
    if args.active == "Sigmoid":
        model = deform_four_by_four(torch.nn.Sigmoid(), args.useattention)
    if args.active == "LeakyReLU":
        model = deform_four_by_four(torch.nn.LeakyReLU(), args.useattention)     

    model.to(args.device)

    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    # optimizer
    min_loss = float('inf')
    print(min_loss)
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr = args.lr)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.001)

    # training
    LOGGER.debug('train, begin')

    loss_values_training_save = []
    loss_values_validation_save = []
    loss_values_cham_save = []
    strainlist = []
    stresslist = []
    energylist = []
    maelist = []

    log = logbox.IOStream(str(log_trainningloss_dir) + '/run.log')

    for epoch in tqdm(range(args.epochs), desc="Epochs: "):
        print('当前epochs: {}/{}'.format(epoch + 1, args.epochs))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

        running_loss, loss_cham, residual_strain, equilibrium, strain_energy, ave_mae = train_1(model, trainloader, optimizer, args.device, args.surfacechamfer, args, log, YM_extracted_rows, PR_extracted_row)

        is_best = running_loss < min_loss # running_loss
        min_loss = min(running_loss, min_loss) # running_loss

        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'optimizer': optimizer.state_dict(), }
        if is_best:
            save_checkpoint(snap, checkpoints_dir, 'snap_best')
            save_checkpoint(model.state_dict(), checkpoints_dir, 'model_best')
        save_checkpoint(snap, checkpoints_dir, 'snap_last')
        save_checkpoint(model.state_dict(), checkpoints_dir, 'model_last')

        loss_values_training_save.append(running_loss) #training loss
        loss_values_cham_save.append(float(loss_cham))
        strainlist.append(float(residual_strain))
        stresslist.append(float(equilibrium))
        energylist.append(float(strain_energy))
        maelist.append(ave_mae)

        with open(log_trainningloss_dir/'training.txt', 'w') as fp:
            for item in loss_values_training_save:
                # write each item on a new line
                fp.write("%s\n" % item)

        with open(log_trainningloss_dir/'vali.txt', 'w') as fp:
            for item in loss_values_validation_save:
                # write each item on a new line
                fp.write("%s\n" % item)

        with open(log_trainningloss_dir/'cham.txt', 'w') as fp:
            for item in loss_values_cham_save:
                # write each item on a new line
                fp.write("%s\n" % item)    

        with open(log_trainningloss_dir/'strainlist.txt', 'w') as fp:
            for item in strainlist:
                # write each item on a new line
                fp.write("%s\n" % item)          
        with open(log_trainningloss_dir/'stresslist.txt', 'w') as fp:
            for item in stresslist:
                # write each item on a new line
                fp.write("%s\n" % item)    
        with open(log_trainningloss_dir/'energylist.txt', 'w') as fp:
            for item in energylist:
                # write each item on a new line
                fp.write("%s\n" % item)                                                          
        if epoch%20==0:
            imaging(loss_values_training_save, str(image_dir)+ '/training', "training")
            imaging(loss_values_validation_save, str(image_dir)+ '/vali', "vali")
            imaging(loss_values_cham_save, str(image_dir)+ '/cham', "chamfer")
            imaging(strainlist, str(image_dir)+'/strain', "strain")
            imaging(stresslist, str(image_dir)+'/stress', "stress")
            imaging(energylist, str(image_dir)+ '/energy', "energy")
    LOGGER.debug('train, end')

def imaging(list_data, path, label):
    x = list(range(len(list_data)))
    plt.figure(figsize=(10, 5))
    plt.plot(x, list_data, marker='o', color='b', label= label +'Loss')
    plt.title(label + 'Loss')
    plt.xlabel('Index')
    plt.ylabel(label + 'Loss Value')
    xticks = [i for i in x if i % 100 == 0]  # 每 100 个索引
    plt.xticks(xticks)
    plt.grid()
    plt.legend()
    plt.savefig(path+'.png', dpi=300)
    plt.close()  


def get_sample(data_dir):
    dirlist = []
    files = [f for f in os.listdir(data_dir) if f.endswith('patient.npy')]
    files.sort(key=lambda x: int(x.split('patient')[0]))
    for file in files:
        file_path = os.path.join(data_dir, file)
        dirlist.append(file_path)
        data = np.load(file_path)  # 读取.npy文件
        print(f"读取文件: {file}, 数据形状: {data.shape}")

    return dirlist

def extract_numbers_from_filenames(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    files.sort(key=lambda x: int(x[:-4]))
    
    numbers = []
    for file in files:
        number = int(file[:-4])
        numbers.append(number)

    return numbers

def main(args):
    set_random_seed(args.seed)
    samples_path = get_sample(args.traindata_path)

    YM = np.load(args.YM)
    PR = np.load(args.PR)

    numbers = extract_numbers_from_filenames(args.traindata_yp_path)

    YM_extracted_rows = YM[numbers]
    PR_extracted_rows = PR[numbers]

    Train_SAMPLES = []
    for index in range(0, len(samples_path)):
        current_sample_path = samples_path[index]
        current_sample = np.load(current_sample_path)
        Train_SAMPLES.append(current_sample)

    trainset = prostateset_v2(Train_SAMPLES)
    run(args, trainset, YM_extracted_rows, PR_extracted_rows)

if __name__ == '__main__':
    print('The Main Function is Running')
    ARGS = options()
    print(ARGS.dim_k)
    print(ARGS.device)
    main(ARGS)