"""
Test models for DDIR.
"""

# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
from sklearn.metrics import mean_squared_error, r2_score
import nibabel as nib
# project
sys.path.append('../ext/medipy-lib')
sys.path.append('../ext/pynd-lib')
import pynd
import medipy
import networks
# import util
from medipy.metrics import dice
import data as datagenerators
import SimpleITK as sitk
import nibabel as nib 

def flow_jacdet(flow):

    vol_size = flow.shape[:-1]
    grid = np.stack(pynd.ndutils.volsize2ndgrid(vol_size), len(vol_size))  
    J = np.gradient(flow + grid)

    dx = J[0]
    dy = J[1]
    dz = J[2]

    Jdet0 = dx[:,:,:,0] * (dy[:,:,:,1] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,1])
    Jdet1 = dx[:,:,:,1] * (dy[:,:,:,0] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,0])
    Jdet2 = dx[:,:,:,2] * (dy[:,:,:,0] * dz[:,:,:,1] - dy[:,:,:,1] * dz[:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    Jaco_num = np.sum(Jdet<=0)
    ratio = Jaco_num/(128*128*32*3)
    
    return Jdet,Jaco_num,ratio



def computeQualityMeasures(lP,lT):
    quality=dict()
    #print('123453454',lP.shape)
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelPred.SetSpacing([3.15,1.5,1.5])
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    labelTrue.SetSpacing([3.15,1.5,1.5,3.15])
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.8,labelPred>0.8)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()
 
    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.1,labelPred>0.1)
    quality["dice"]=dicecomputer.GetDiceCoefficient()
 
    return quality

labels = {1: 'lv',
          2: 'lvm',
          3: 'rv'}




def ejection_prediction(seg_ed,seg_es,pixel_volume):
    density = 1.05
    LVEDV = np.sum(seg_ed[:, :, :] == 1) * pixel_volume
    LVM = np.sum(seg_ed[:, :, :] == 2) * pixel_volume * density
    RVEDV = np.sum(seg_ed[:, :, :] == 3) * pixel_volume
                       
    LVESV = np.sum(seg_es[:, :, :] == 1) * pixel_volume
    RVESV = np.sum(seg_es[:, :, :] == 3) * pixel_volume  
    
    LVSV =   LVEDV -  LVESV 
    LVEF =   LVSV / LVEDV * 100
    RVSV = RVEDV - RVESV
    RVEF = RVSV / RVEDV * 100
    return LVEDV, LVESV, LVSV, LVEF, LVM, RVEDV, RVESV, RVSV, RVEF

seg1=tf.placeholder(tf.float32,shape=(None, 128,128,128))
seg2=tf.placeholder(tf.float32,shape=(None, 128,128,128))
DICE,_ = mask_metrics(seg1, seg2)

data_dir = "/usr/not-backed-up/scxc/COATT/testing_mask/"
test_names = glob.glob(os.path.join(data_dir, '*.nii.gz'))   
n_batches = len(test_names)
print(n_batches)

def test(gpu_id, model_dir, iter_num, 
         compute_type = 'GPU',  # GPU or CPU
         vol_size=(128,128,32),
         nf_enc = [16,32,32,32],
         nf_dec = [32,32,32,32,16,3],
         save_file=None):

    # GPU handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        # load trained model
        net = networks.DDIR(vol_size, nf_enc, nf_dec, use_miccai_int=False, indexing='ij')  
        net.load_weights(os.path.join(model_dir, str(iter_num) + '.h5'))

        # diffeomorphic flow output model
        diff_net0 = keras.models.Model(net.inputs, net.get_layer('diffflow0').output)
        diff_net1 = keras.models.Model(net.inputs, net.get_layer('diffflow1').output)
        diff_net2 = keras.models.Model(net.inputs, net.get_layer('diffflow2').output)
        diff_net3 = keras.models.Model(net.inputs, net.get_layer('diffflow3').output)
        diff_net = keras.models.Model(net.inputs, net.get_layer('diffflow').output)

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

    # if CPU, prepare grid
    if compute_type == 'CPU':
        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)
    
    # prepare a matrix of dice values
    dice0 = []
    dice1 = []
    dice_ori = []
    dice_ori1 = []
    indices_gt = []
    indices_pre = []
    hd_ori = []
    hd = []
    elastix_dice = []
    elastix_dice1 = []
    elastix_hd = []
    RMSE = []
    RMSE_ori = []
    RMSE_tr = []
    nag_jec_num = []
    lv_nag_jec_num = []
    lvm_nag_jec_num = []
    rv_nag_jec_num = []
    nag_jec_ratio = []
    for k in range(2*n_batches):
        # get data
        if k!=708:
            continue
        if k >= n_batches:
            seg_name = test_names[k-n_batches]
            flag = 1
            X_vol, X_seg, atlas_vol, atlas_seg, affine,pixel_volume = datagenerators.test_data(seg_name,flag)
        else:
            seg_name = test_names[k]
            flag = 0
            X_vol, X_seg, atlas_vol, atlas_seg, affine,pixel_volume = datagenerators.test_data(seg_name,flag)
        dice_ori_vals, _ = dice(X_seg[0,:,:,:,0], atlas_seg[0,:,:,:,0], labels=None, nargout=2)
        if(len(dice_ori_vals)<3):
            print('=========',k,seg_name,dice_ori_vals)
            continue
        dice_ori1.append(dice_ori_vals[:3])
        
        dice_ori_vals = computeQualityMeasures(X_seg[0,:,:,:,0], atlas_seg[0,:,:,:,0])
        dice_ori.append(dice_ori_vals['dice'])

        if flag:
            indices = ejection_prediction(X_seg[0,:,:,:,0], atlas_seg[0,:,:,:,0],pixel_volume)
        else:
            indices = ejection_prediction(atlas_seg[0,:,:,:,0], X_seg[0,:,:,:,0],pixel_volume)
        indices_gt.append(indices)
        hd_ori.append(dice_ori_vals['Hausdorff'])
        rmse_ori = np.sqrt(((X_vol[0,:,:,:,0]- atlas_vol[0,...,0])) ** 2).mean()
        RMSE_ori.append(rmse_ori)

        with tf.device(gpu):
            pred = diff_net.predict([X_vol, atlas_vol,X_seg, atlas_seg])
            flow0 = diff_net0.predict([X_vol, atlas_vol,X_seg, atlas_seg])
            flow1 = diff_net1.predict([X_vol, atlas_vol,X_seg, atlas_seg])
            flow2 = diff_net2.predict([X_vol, atlas_vol,X_seg, atlas_seg])
            flow3 = diff_net3.predict([X_vol, atlas_vol,X_seg, atlas_seg])

        # Warp segments with flow
        if compute_type == 'CPU':
            flow = pred[0, :, :, :, :]
            warped_image = util.warp_seg(X_vol, flow, grid=grid, xx=xx, yy=yy, zz=zz)
            warped_seg = util.warp_seg(X_seg, flow, grid=grid, xx=xx, yy=yy, zz=zz)

        else:  # GPU
            warped_image = nn_trf_model.predict([X_vol, pred])[0,...,0]
            warped_seg = nn_trf_model.predict([X_seg, pred])[0,...,0]

        dice_vals, _ = dice(warped_seg, atlas_seg[0,:,:,:,0], labels=None, nargout=2)
        dice1.append(dice_vals[:3])
        print('=========',k,dice_vals[:3])

        if flag:
            indices = ejection_prediction(X_seg[0,:,:,:,0], warped_seg,pixel_volume)
        else:
            indices = ejection_prediction(warped_seg,X_seg[0,:,:,:,0],pixel_volume)
        indices_pre.append(indices)

        dice_vals = computeQualityMeasures(warped_seg, atlas_seg[0,:,:,:,0])
        dice0.append(dice_vals['dice'])

        hd.append(dice_vals['Hausdorff'])
        rmse = np.sqrt(((warped_image- atlas_vol[0,...,0])) ** 2).mean()
        RMSE.append(rmse)
        
        Jdet,Jaco_num,ratio = flow_jacdet(pred[0, :, :, :, :])
        nag_jec_num.append(Jaco_num)
        nag_jec_ratio.append(ratio)
        lv_nag_jec_num.append(flow_jacdet(flow1[0, :, :, :, :])[1])
        lvm_nag_jec_num.append(flow_jacdet(flow2[0, :, :, :, :])[1])
        rv_nag_jec_num.append(flow_jacdet(flow3[0, :, :, :, :])[1])
        

        ddf_magnitude = pred[0,:,:,:,:]
        ddf_magnitude = np.sum(np.square(ddf_magnitude),3)
        train_tensor = compute_strain(pred[0,:,:,:,:])
        strain_lvm = compute_strain(flow2[0,:,:,:,:])
        if k==0:
            img = np.array(X_vol[0,:,:,:,0])
            mask = np.array(X_seg[0,:,:,:,0])
            img_bk = (mask==0)*img
            img_lv = (mask==1)*img
            img_lvm = (mask==2)*img
            img_rv = (mask==3)*img
            img_bk = sitk.GetImageFromArray(img_bk)
            sitk.WriteImage(img_bk, "results/img_bk"+str(k)+".nii.gz")
            img_lv = sitk.GetImageFromArray(img_lv)
            sitk.WriteImage(img_lv, "results/img_lv"+str(k)+".nii.gz")
            img_lvm = sitk.GetImageFromArray(img_lvm)
            sitk.WriteImage(img_lvm, "results/img_lvm"+str(k)+".nii.gz")
            img_rv = sitk.GetImageFromArray(img_rv)
            sitk.WriteImage(img_rv, "results/img_rv"+str(k)+".nii.gz")

            img = np.array(atlas_vol[0,:,:,:,0])
            mask = np.array(atlas_seg[0,:,:,:,0])
            img_bk = (mask==0)*img
            img_lv = (mask==1)*img
            img_lvm = (mask==2)*img
            img_rv = (mask==3)*img
            img_bk = sitk.GetImageFromArray(img_bk)
            sitk.WriteImage(img_bk, "results/fix_bk"+str(k)+".nii.gz")
            img_lv = sitk.GetImageFromArray(img_lv)
            sitk.WriteImage(img_lv, "results/fix_lv"+str(k)+".nii.gz")
            img_lvm = sitk.GetImageFromArray(img_lvm)
            sitk.WriteImage(img_lvm, "results/fix_lvm"+str(k)+".nii.gz")
            img_rv = sitk.GetImageFromArray(img_rv)
            sitk.WriteImage(img_rv, "results/fix_rv"+str(k)+".nii.gz")

            if 0: #using nib
                moving = nib.Nifti1Image(X_vol[0,:,:,:,0],affine)
                nib.save(moving, "results/moving"+str(k)+".nii.gz")
                fixed = nib.Nifti1Image(atlas_vol[0,:,:,:,0],affine)
                nib.save(fixed, "results/fixed"+str(k)+".nii.gz")
                warped = nib.Nifti1Image(warped_image,affine)
                nib.save(warped, "results/wapredimage"+str(k)+".nii.gz")
                field = nib.Nifti1Image(pred[0,:,:,:,:],affine)
                nib.save(field, "results/deformation"+str(k)+".nii.gz")
                out = sitk.GetImageFromArray(pred[0,:,:,:,:])
                Jdet = nib.Nifti1Image(Jdet,affine)
                nib.save(Jdet, "results/Jdet"+str(k)+".nii.gz")
                field = nib.Nifti1Image(flow0[0,:,:,:,:],affine)
                nib.save(field, "results/ddf_bk"+str(k)+".nii.gz")
                field = nib.Nifti1Image(flow1[0,:,:,:,:],affine)
                nib.save(field, "results/ddf_lv"+str(k)+".nii.gz")
                field = nib.Nifti1Image(flow2[0,:,:,:,:],affine)
                nib.save(field, "results/ddf_lvm"+str(k)+".nii.gz")
                field = nib.Nifti1Image(flow3[0,:,:,:,:],affine)
                nib.save(field, "results/ddf_rv"+str(k)+".nii.gz")
            else:#using sitk
                moving = sitk.GetImageFromArray(X_vol[0,:,:,:,0])
                sitk.WriteImage(moving, "results/moving"+str(k)+".nii.gz")
                moving_seg = sitk.GetImageFromArray(X_seg[0,:,:,:,0])
                sitk.WriteImage(moving_seg, "results/moving_seg"+str(k)+".nii.gz")
                fixed_seg = sitk.GetImageFromArray(atlas_seg[0,:,:,:,0])
                fixed_seg.SetSpacing([3.15,1.5,1.5])
                sitk.WriteImage(fixed_seg, "results/fixed_seg"+str(k)+".nii.gz")
                fixed = sitk.GetImageFromArray(atlas_vol[0,:,:,:,0])
                sitk.WriteImage(fixed, "results/fixed"+str(k)+".nii.gz")
                warped = sitk.GetImageFromArray(warped_image)
                sitk.WriteImage(warped, "results/wapredimage"+str(k)+".nii.gz")
                field = sitk.GetImageFromArray(pred[0,:,:,:,:])
                sitk.WriteImage(field, "results/deformation"+str(k)+".nii.gz")
                field_mag = sitk.GetImageFromArray(ddf_magnitude)
                sitk.WriteImage(field_mag, "results/ddf_mag"+str(k)+".nii.gz")
                Jdet = sitk.GetImageFromArray(Jdet)
                sitk.WriteImage(Jdet, "results/Jdet"+str(k)+".nii.gz")
                strain = sitk.GetImageFromArray(train_tensor)
                sitk.WriteImage(strain, "results/strain"+str(k)+".nii.gz")
                strain_lvm = sitk.GetImageFromArray(strain_lvm)
                sitk.WriteImage(strain_lvm, "results/strain_lvm"+str(k)+".nii.gz")
                field = sitk.GetImageFromArray(flow0[0,:,:,:,:])
                sitk.WriteImage(field, "results/ddf_bk"+str(k)+".nii.gz")
                field = sitk.GetImageFromArray(flow1[0,:,:,:,:])
                sitk.WriteImage(field, "results/ddf_lv"+str(k)+".nii.gz")
                field = sitk.GetImageFromArray(flow2[0,:,:,:,:])
                sitk.WriteImage(field, "results/ddf_lvm"+str(k)+".nii.gz")
                field = sitk.GetImageFromArray(flow3[0,:,:,:,:])
                sitk.WriteImage(field, "results/ddf_rv"+str(k)+".nii.gz")
        
    dice_ori1 = np.array(dice_ori1)
    dice1 = np.array(dice1)
    elastix_dice1 = np.array(elastix_dice1)
    indices_pre = np.array(indices_pre)
    indices_gt = np.array(indices_gt)
    print(dice_ori1.shape,dice1.shape)
    print('vmdice before registration:',np.mean(dice_ori1[:,0]),np.std(dice_ori1[:,0]),'||',np.mean(dice_ori1[:,1]),np.std(dice_ori1[:,1]),'||',np.mean(dice_ori1[:,2]),np.std(dice_ori1[:,2]),'||',np.mean(dice_ori1),np.std(np.mean(dice_ori1,1)))
    print('vmdice after registration:',np.mean(dice1[:,0]),np.std(dice1[:,0]),'||',np.mean(dice1[:,1]),np.std(dice1[:,1]),'||', np.mean(dice1[:,2]),np.std(dice1[:,2]),'||',np.mean(dice1),np.std(np.mean(dice1,1)))
    print('original dice mean:',np.mean(dice_ori),np.std(dice_ori))
    print('original hd mean:',np.mean(hd_ori),np.std(hd_ori))
    print('dice mean:',np.mean(dice0),np.std(dice0))
    print('hd mean:',np.mean(hd),np.std(hd))
    print('RMSE_ori:',np.mean(RMSE_ori),np.std(RMSE_ori))
    print('RMSE:',np.mean(RMSE),np.std(RMSE))
    print('indices_gt:',np.mean(indices_gt),np.std(indices_gt))
    print('indices_pre:',np.mean(indices_pre),np.std(indices_pre))
    print('nagetive jecobean determinant:', np.mean(nag_jec_num), np.std(nag_jec_num), np.mean(nag_jec_ratio))
    print('lv lvm rv nagetive jecobean determinant:', np.mean(lv_nag_jec_num), np.std(lv_nag_jec_num),'||', np.mean(lvm_nag_jec_num), np.std(lvm_nag_jec_num),'||', np.mean(rv_nag_jec_num), np.std(rv_nag_jec_num))
    import h5py
    with h5py.File('results/DDIR.h5', 'w') as hf:
        hf.create_dataset("carnet_dice",  data=np.mean(dice1,1))
        hf.create_dataset("carnet_hd",  data=hd)
        hf.create_dataset("carnet_lv_dice",  data=dice1[:,0])
        hf.create_dataset("carnet_lvm_dice",  data=dice1[:,1])
        hf.create_dataset("carnet_rv_dice",  data=dice1[:,2])
        hf.create_dataset("indices_pre",  data=indices_pre)
        hf.create_dataset("indices_gt",  data=indices_gt)
    
    print('elastix dice mean:',np.mean(elastix_dice),np.std(elastix_dice))
    print('elastix hd mean:',np.mean(elastix_hd),np.std(elastix_hd))
    print('elastix dice mean:',np.mean(elastix_dice1[:,0]),np.std(elastix_dice1[:,0]),'||',np.mean(elastix_dice1[:,1]),np.std(elastix_dice1[:,1]),'||',np.mean(elastix_dice1[:,2]),np.std(elastix_dice1[:,2]),'||', np.mean(elastix_dice1),np.std(np.mean(elastix_dice1,1)))
    print('tr RMSE:',np.mean(RMSE_tr),np.std(RMSE_tr))

if __name__ == "__main__":
    """
    assuming the model is model_dir/iter_num.h5
    python test.py gpu_id model_dir iter_num
    """
    test(sys.argv[1], sys.argv[2], sys.argv[3])