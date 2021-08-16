"""
data generators for DDIR.
"""

import os, sys
import numpy as np
from skimage import transform
import nibabel as nib  
import SimpleITK as sitk
import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy 


def train_data(vol_names, batch_size=1, return_segs=False, seg_dir=None, np_var='vol_data'):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """
    zeros = None
    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        Y_data = []
        X_SEGdata = []
        Y_SEGdata = []

        for idx in idxes:
            
            img_path = vol_names[idx].replace('training_mask','processed_img1')
            seg_path = vol_names[idx].replace('training_mask','processed_mask1')
            #seg_path = vol_names[idx]
            X, Y = load_volfile_ukbb(img_path, np_var=np_var) #exp
            X_seg, Y_seg = load_segfile_ukbb(seg_path, np_var)
            X = X[np.newaxis, ..., np.newaxis]
            Y = Y[np.newaxis, ..., np.newaxis]
            X_seg = X_seg[np.newaxis, ..., np.newaxis]
            Y_seg = Y_seg[np.newaxis, ..., np.newaxis]

           
            if np.random.choice([0,1])==1:
                X_data.append(X)
                Y_data.append(Y)
                X_SEGdata.append(X_seg)
                Y_SEGdata.append(Y_seg)
                #print(np.random.choice([0,1]))
            else:
                X_data.append(Y)
                Y_data.append(X)
                X_SEGdata.append(Y_seg)
                Y_SEGdata.append(X_seg)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0),np.concatenate(Y_data, 0),np.concatenate(X_SEGdata, 0),np.concatenate(Y_SEGdata, 0)]
        else:
            return_vals = [X_data[0],Y_data[0],X_SEGdata[0],Y_SEGdata[0]]

        if zeros is None:
            volshape = X_data[0].shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))

        #print(return_vals[1].shape,return_vals[2].shape,return_vals[3].shape)
        yield (return_vals,[np.concatenate(Y_data, 0), np.concatenate(Y_SEGdata, 0),zeros,zeros,zeros,zeros])

def test_data(seg_name, ed2es):
    """
    load a specific volume and segmentation

    np_var: specify the name of the variable in numpy files, if your data is stored in 
        npz files. default to 'vol_data'
    """
    np_var='vol_data'
    if 'validation_mask' in seg_name:
        vol_name = seg_name.replace('validation_mask','processed_img')
    else:
        vol_name = seg_name.replace('testing_mask','processed_img')
    if ed2es:
        X, Y, affine,pixel_volume = load_volfile_ukbb1(vol_name, np_var)
        X_seg, Y_seg = load_segfile_ukbb(seg_name, np_var)
    else:
        Y, X, affine,pixel_volume = load_volfile_ukbb1(vol_name, np_var)
        Y_seg, X_seg = load_segfile_ukbb(seg_name, np_var)        
    X = X[np.newaxis, ..., np.newaxis]
    Y = Y[np.newaxis, ..., np.newaxis]
    X_seg = X_seg[np.newaxis, ..., np.newaxis]
    Y_seg = Y_seg[np.newaxis, ..., np.newaxis]

    return_vals = [X]
    return_vals.append(X_seg)
    return_vals.append(Y)
    return_vals.append(Y_seg)
    return_vals.append(affine)
    return_vals.append(pixel_volume)
    return tuple(return_vals)


def load_volfile_ukbb(datafile, np_var='vol_data'):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        #if 'nibabel' not in sys.modules:
        #    try :
        #        import nibabel as nib  
        #    except:
        #        print('Failed to import nibabel. need nibabel library for these data file types.')
        
        #X = nib.load(datafile).get_data()
        data = nib.load(datafile).get_data()
        #data = sitk.ReadImage(datafile)#.get_data()
        #data = sitk.GetArrayViewFromImage(data)
        
        if(len(data.shape)!=4):
            print(datafile,data.shape)
        X1 = data[:,:,:,0]
        X2 = data[:,:,:,1]

        X1 = (X1-X1.min())/(X1.max()-X1.min())
        X2 = (X2-X2.min())/(X2.max()-X2.min())
        #print(data.shape)
        if (X1.shape[2]<32):
            X1 = np.pad(X1, ((0, 0), (0, 0), (0, 32-X1.shape[2])), 'constant')
        if (X2.shape[2]<32):
            X2 = np.pad(X2, ((0, 0), (0, 0), (0, 32-X2.shape[2])), 'constant')
        #print('2345',data.shape,datafile,X1.shape,X2.shape)
    else: # npz
        if np_var is None:
            np_var = 'vol_data'
        X = np.load(datafile)[np_var]

    return X1,X2

def load_volfile_ukbb1(datafile, np_var='vol_data'):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        #if 'nibabel' not in sys.modules:
        #    try :
        #        import nibabel as nib  
        #    except:
        #        print('Failed to import nibabel. need nibabel library for these data file types.')
        
        #X = nib.load(datafile).get_data()
        image = nib.load(datafile)
        pixdim = image.header['pixdim'][1:4]
        pixel_volume = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3
        
        affine = image.affine
        data = image.get_data()
        #data = sitk.ReadImage(datafile)#.get_data()
        #data = sitk.GetArrayViewFromImage(data)
        
        if(len(data.shape)!=4):
            print(datafile,data.shape)
        X1 = data[:,:,:,0]
        X2 = data[:,:,:,1]

        X1 = (X1-X1.min())/(X1.max()-X1.min())
        X2 = (X2-X2.min())/(X2.max()-X2.min())
        #print(data.shape)
        if (X1.shape[2]<32):
            X1 = np.pad(X1, ((0, 0), (0, 0), (0, 32-X1.shape[2])), 'constant')
        if (X2.shape[2]<32):
            X2 = np.pad(X2, ((0, 0), (0, 0), (0, 32-X2.shape[2])), 'constant')
        #print('2345',data.shape,datafile,X1.shape,X2.shape)
    else: # npz
        if np_var is None:
            np_var = 'vol_data'
        X = np.load(datafile)[np_var]

    return X1,X2,affine,pixel_volume


def load_segfile_ukbb(datafile, np_var='vol_data'):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        #if 'nibabel' not in sys.modules:
        #    try :
        #        import nibabel as nib  
        #    except:
        #        print('Failed to import nibabel. need nibabel library for these data file types.')

        #X = nib.load(datafile).get_data()
        image = nib.load(datafile)
        affine = image.affine
        data = image.get_data()
        X1 = data[:,:,:,0]
        X2 = data[:,:,:,1]
        
        if (X1.shape[2]<32):
            X1 = np.pad(X1, ((0, 0), (0, 0), (0, 32-X1.shape[2])), 'constant')
        if (X2.shape[2]<32):
            X2 = np.pad(X2, ((0, 0), (0, 0), (0, 32-X2.shape[2])), 'constant')
            #print(X.shape)
        
        #X = transform.resize(X,( 128, 128,32))
    else: # npz
        if np_var is None:
            np_var = 'vol_data'
        X = np.load(datafile)[np_var]

    return X1,X2
