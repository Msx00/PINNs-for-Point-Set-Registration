import torch
import numpy as np
import argparse
import torch.utils.data
import logging
import os
from tqdm import tqdm
from train import get_sample, options, prostateset_v2, deform_four_by_four,set_random_seed
from scipy.stats import wasserstein_distance
from savetxt import savetxt
from redeal import dataset

def chamfer_loss(x, y, ps):
    A = x
    B = y
    r = torch.sum(A * A, dim=2).unsqueeze(-1)
    r1 = torch.sum(B * B, dim=2).unsqueeze(-1)
    t = r.repeat(1, 1, ps) - 2 * torch.bmm(A, B.permute(0, 2, 1)) + r1.permute(0, 2, 1).repeat(1, ps, 1)
    d1, _ = t.min(dim=2)
    d2, _ = t.min(dim=1)
    unsquared_d1 = torch.sqrt(d1 + 1e-10)
    unsquared_d2 = torch.sqrt(d2 + 1e-10)
    sum_d1 = unsquared_d1.sum(dim=1)
    sum_d2 = unsquared_d2.sum(dim=1)
    chamfer_distance = 0.5 * (sum_d1 / ps + sum_d2 / ps)
    return float(chamfer_distance.mean())


def chamfer_loss2(x,y,ps):
    A= x 
    r=torch.sum(A*A,dim=2) 
    r=r.unsqueeze(-1) 
    B= y                                                 
    r1=torch.sum(B*B,dim=2) 
    r1=r1.unsqueeze(-1)
    t=(r.repeat(1,1,ps) -2*torch.bmm(A,B.permute(0,2,1)) + r1.permute(0, 2, 1).repeat(1,ps,1))
    d1,_=t.min(dim=1)
    d2,_=t.min(dim=2)
    ls=(d1+d2)/2
    return ls.mean()


def load_us_centroid_distance(abs_dir):
    pointsets_list, mr_centroid, us_centroid, mr_furthest_distance, us_furthest_distance = dataset()
    mr_gt = []
    files = os.listdir(abs_dir)
    gt_npy_files = [f for f in files if f.endswith('point_set_mri_deformed_gt.npy')]
    gt_npy_files.sort(key=lambda x: int(x.split('_')[0]))
    for file in gt_npy_files:
        file_path = os.path.join(abs_dir, file)
        data = np.load(file_path)
        mr_gt.append(data)
        print(f'Read {file_path, data.shape}')

    return np.array(mr_centroid), np.array(us_centroid), np.array(mr_furthest_distance), np.array(us_furthest_distance), np.array(mr_gt)


def HausdorffDistance(A, B):
    A = np.array(A)
    B = np.array(B)
    distances = np.linalg.norm(A[:, np.newaxis] - B, axis=2)
    min_distances = np.min(distances, axis=1)
    data1 = np.max(min_distances)
    distances = np.linalg.norm(B[:, np.newaxis] - A, axis=2)
    min_distances = np.min(distances, axis=1)
    data2 = np.max(min_distances)
    return max(data1, data2)


def compute_rmse(pc1, pc2):
    if pc1.shape != pc2.shape:
        raise ValueError("Point clouds must have the same shape")
    distances = np.linalg.norm(pc1 - pc2, axis=1)
    rmse = np.sqrt(np.mean(distances**2))
    return rmse


def emd(mri_deformed_original_denormalise, us_deformed_original_denormalise):
    emd_distance1 = wasserstein_distance(mri_deformed_original_denormalise[:,0], us_deformed_original_denormalise[:,0])
    emd_distance2 = wasserstein_distance(mri_deformed_original_denormalise[:,1], us_deformed_original_denormalise[:,1])
    emd_distance3 = wasserstein_distance(mri_deformed_original_denormalise[:,2], us_deformed_original_denormalise[:,2])
    return (emd_distance1 + emd_distance2 + emd_distance3)/3


def compute_mae(pc1, pc2):
    if pc1.shape != pc2.shape:
        raise ValueError("Point clouds must have the same shape")
    absolute_errors = np.abs(pc1 - pc2)
    mae = np.mean(absolute_errors)
    return mae


def load_model(pth_dir, device, active_founc, useattention):
    checkpoint = torch.load(pth_dir)
    model = deform_four_by_four(active_founc, useattention)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    print("have load model")
    return model


def return_deform_list(model, testdataset, device):
    collect_deform_list = []
    collect_testdata_list = []
    jacobianlist = []
    for i, data in enumerate(testdataset):
        data = data.float().to(device)
        data.requires_grad = True
        deformed, displacements, strain = model(data)

        if i == 21:
            displacements_save = displacements.permute(0, 2, 1)
            deformed_mri_array_11 = displacements_save[0, :, :] + data[0, 0, :, :]
            us_array_11 = data[0, 1, :, :]
        deform_cpu = deformed.cpu().detach().numpy()
        data_cpu = data.cpu().detach().numpy()
        jacobian = 0

        collect_deform_list.append(deform_cpu.tolist())
        collect_testdata_list.append(data_cpu.tolist())
        jacobianlist.append(jacobian)
    return collect_deform_list, collect_testdata_list, jacobianlist



def return_deform(model, singledata, device):
    with torch.no_grad():
        print("singledata.shape", singledata.shape)
        
        temp_deform = model(singledata)
        deform, displacement = temp_deform

        jacobian = 0

        deform_np = deform.cpu().detach().numpy()

        temp_deform_array = np.array(deform_np)#.cpu().detach().numpy()
        singledata_array = singledata.cpu().detach().numpy()#//singledata_array maybe same with  testdataset[i]
        return temp_deform_array, singledata_array, jacobian


def de_normalise_pointcloud(points, centroid, furthest_distance):
    points = points.astype(float)
    centroid = centroid.astype(float)
    furthest_distance = furthest_distance.astype(float)
    points *=furthest_distance
    points +=centroid
    return points


def plot():
    return None


def return_rmse(error, mr_centroid_array, us_centroid_array, mr_furthest_distance_array, us_furthest_distance_array, collect_deform_array, collect_testdata_array, gt_arrays):#//计算TRE一种采用未替换的计算，一种采用替换了的计算
    ave_rmse = 0
    ave_cd = 0
    ave_hd = 0
    ave_emd = 0
    ave_mae = 0
    rmselist = []
    cdlist = []
    emdlist = []
    hdlist = []
    maelist = []
    for num in range(0, gt_arrays.shape[0]):
        print("collect_testdata_array", collect_testdata_array.shape, mr_centroid_array.shape, mr_furthest_distance_array.shape)

        mr_source = (de_normalise_pointcloud(collect_testdata_array[num, 0, 0, :, :], np.array(mr_centroid_array[num]), np.array(mr_furthest_distance_array[num])))
        mri_deformed_original_denormalise = (de_normalise_pointcloud(collect_deform_array[num, 0], np.array(us_centroid_array[num]), np.array(us_furthest_distance_array[num])))
        us_deformed_original_denormalise = (de_normalise_pointcloud(collect_testdata_array[num, 0, 1], np.array(us_centroid_array[num]), np.array(us_furthest_distance_array[num])))

        if error ==0:
            rmse = compute_rmse(mri_deformed_original_denormalise, us_deformed_original_denormalise)

            if not os.path.exists("./WARPED-Linear"):
                os.mkdir("./WARPED-Linear/")
            np.savetxt("./WARPED-Linear/" + str(num) + "warped_MR.txt", mri_deformed_original_denormalise)

            ave_rmse = ave_rmse + rmse
            rmselist.append(float(rmse))

        if error == 1:
            chamfer_dist = chamfer_loss(torch.from_numpy(mri_deformed_original_denormalise[0:1024,:].reshape(1, 1024, 3)), torch.from_numpy(us_deformed_original_denormalise[0:1024,:].reshape(1, 1024, 3)),1024)
            ave_cd = ave_cd + chamfer_dist
            cdlist.append(chamfer_dist)

        if error == 2:
            emd_distance = emd(mri_deformed_original_denormalise, us_deformed_original_denormalise)
            ave_emd = ave_emd + emd_distance
            emdlist.append(emd_distance)
            
        if error == 3:
            hd = HausdorffDistance(mri_deformed_original_denormalise, us_deformed_original_denormalise)
            ave_hd = ave_hd + hd

        if error == 4:
            mae = compute_mae(mri_deformed_original_denormalise, us_deformed_original_denormalise)
            maelist.append(mae)
            ave_mae = ave_mae + mae

        if error == 5:
            rmse = compute_rmse(mri_deformed_original_denormalise, us_deformed_original_denormalise)

    return_data = 0          
    if error == 0:
        return_data = ave_rmse
    if error == 1:
        return_data = ave_cd
    if error == 2:
        return_data = ave_emd
    if error == 3:
        return_data = ave_hd
    if error == 4:
        return_data = ave_mae        
    
    return return_data, rmselist, cdlist, emdlist, hdlist, maelist



def main(error, abs_dir, model, testdataset, device, log_str, active_founc,useattention):
    set_random_seed(3407)

    mr_centroid_array, us_centroid_array, mr_furthest_distance_array, us_furthest_distance_array, gt_array = load_us_centroid_distance(abs_dir)

    collect_deform_list, collect_testdata_list, jacobianlist = return_deform_list(model, testdataset, device)
    collect_deform_array = np.array(collect_deform_list)    
    collect_testdata_array = np.array(collect_testdata_list)

    return_data, trelist, cdlist, emdlist, hdlist, maelist = return_rmse(error, mr_centroid_array, us_centroid_array, mr_furthest_distance_array, us_furthest_distance_array,
    collect_deform_array, collect_testdata_array, gt_array)

    return return_data, trelist, cdlist, emdlist, hdlist, jacobianlist, maelist



def load_testdataset(args):
    pointsets_list, mr_centroid, us_centroid, mr_furthest_distance, us_furthest_distance = dataset()
    Train_SAMPLES = pointsets_list

    trainset = prostateset_v2(Train_SAMPLES)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=True)

    return testloader


if __name__ == "__main__":
    total_data = []
    args = options()
    device = args.device
    testdataset = load_testdataset(args)
    print(len(testdataset))
    for seed in tqdm(range(1)):
        seed = 3407
        set_random_seed(seed)
        abs_dir = "./data/gt_centroid_distance"

        for pth_num in range(0, 1):
            i = 0
            if pth_num == 0: 
                print("cd_0.0001elas_2000epochs_leakyrelu_lr0")
                useattention = False
                active_founc = torch.nn.LeakyReLU() # torch.nn.GELU(), torch.nn.Tanh(), torch.nn.Softplus(), torch.nn.ReLU(), torch.nn.ELU(), torch.nn.Sigmoid(), LeakyReLU
                root_dir = "./log/2025-01-11_10-34/"
                pth_dir = root_dir + "checkpoints/checkpoints_snap_best.pth"
                test_dir = root_dir + "testresult"
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)
                model = load_model(pth_dir, device, active_founc, useattention)
            for error in range(0,5):
                i += 1

                return_data, RMSElist, cdlist, emdlist, hdlist, jacobianlist, maelist = main(error, abs_dir, model, testdataset, device, "基于" + pth_dir +"开始计算TRE", active_founc, useattention)                                 
                total_data.append(return_data)
                if i == 5:
                    total_data.append(str(seed) + ": ")

                if error == 0:
                    savetxt(test_dir + "/", "0RMSE", RMSElist)
                if error == 1:
                    savetxt(test_dir + "/", "1cd", cdlist)
                if error == 2:
                    savetxt(test_dir + "/", "2emd", emdlist)
                if error == 3:
                    savetxt(test_dir + "/", "3hd", hdlist)                                                
                if error == 3:
                    savetxt(test_dir + "/", "5jacobian", jacobianlist)
                if error == 4:
                    savetxt(test_dir + "/", "4mae", maelist)


    print("计算结束")
    print(total_data)
