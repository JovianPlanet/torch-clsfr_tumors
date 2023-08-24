import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
import nibabel.processing


class CNN3D_DS(Dataset):

    def __init__(self, config, mode):

        self.config = config
        self.mode   = mode

        tum_dir = ''
        norm_dir = ''
        n = 0

        if self.mode == 'train':
            tum_dir = self.config['data']['brats_train'] 
            norm_dir = self.config['data']['nfbs_train']
            n = self.config['hyperparams']['n_train']

        elif self.mode == 'val':
            tum_dir = self.config['data']['brats_val']
            norm_dir = self.config['data']['nfbs_val']
            n = self.config['hyperparams']['n_val']

        elif self.mode == 'test':
            tum_dir = self.config['data']['brats_test']
            norm_dir = self.config['data']['nfbs_test']
            n = self.config['hyperparams']['n_test']

        self.tum_subs = next(os.walk(tum_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?
        self.norm_subs = next(os.walk(norm_dir))[1]

        self.L = []

        for subject in self.tum_subs[:n]:
            if '355' in subject: continue
            #print(f'\nsujeto: {subject}')
            files = next(os.walk(os.path.join(tum_dir, subject)))[2]
            for file_ in files:
                if 't1.nii' in file_:
                    #print(f'\tfile_: {file_}')
                    mri_path = os.path.join(tum_dir, subject, file_)

            self.L.append([subject, mri_path, 1])

        for subject in self.norm_subs[:n]:
            file_ = 'sub-'+subject+'_ses-NFB3_T1w_brain.nii.gz'
            mri_path = os.path.join(norm_dir, subject, file_)
            self.L.append([subject, mri_path, 0])

        self.df = pd.DataFrame(self.L, columns=['Subject', 'Path MRI', 'Label'])
        self.df = self.df.assign(id=self.df.index.values).sample(frac=1)
        print(f'dataframe: \n{self.df} \n')


    def __len__(self):

        return self.df.shape[0]


    def __getitem__(self, index):

        mri = preprocess(self.df.at[index, 'Path MRI'], self.config, norm=True)
        label = self.df.at[index, 'Label']

        return mri, label


class Cnn2D_Ds(Dataset):

    def __init__(self, config, mode):

        self.config = config
        self.mode   = mode

        tum_dir = ''
        norm_dir = ''
        n = 0

        if self.mode == 'train':
            tum_dir = self.config['data']['brats_train'] 
            norm_dir = self.config['data']['nfbs_train']
            n = self.config['hyperparams']['n_train']

        elif self.mode == 'val':
            tum_dir = self.config['data']['brats_val']
            norm_dir = self.config['data']['nfbs_val']
            n = self.config['hyperparams']['n_val']

        elif self.mode == 'test':
            tum_dir = self.config['data']['brats_test']
            norm_dir = self.config['data']['nfbs_test']
            n = self.config['hyperparams']['n_test']

        self.tum_subs = next(os.walk(tum_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?
        self.norm_subs = next(os.walk(norm_dir))[1]

        self.L = []

        for subject in self.tum_subs[:n]:
            if '355' in subject: continue
            #print(f'\nsujeto: {subject}')
            files = next(os.walk(os.path.join(tum_dir, subject)))[2]
            for file_ in files:
                if 't1.nii' in file_:
                    #print(f'\tfile_: {file_}')
                    mri_path = os.path.join(tum_dir, subject, file_)

                if 'seg.nii' in file_:
                    #print(f'\tfile_: {file_}')
                    label_path = os.path.join(tum_dir, subject, file_)

            l = preprocess(label_path, self.config, norm=True)

            for slice_ in range(self.config['hyperparams']['model_dims'][2]):

                if np.any(l[:, :, slice_]):
                    self.L.append([subject, mri_path, slice_, 1.])
                else:
                    self.L.append([subject, mri_path, slice_, 0.])

        # for subject in self.norm_subs[:n]:
        #     file_ = 'sub-'+subject+'_ses-NFB3_T1w_brain.nii.gz'
        #     mri_path = os.path.join(norm_dir, subject, file_)
        #     for slice_ in range(self.config['hyperparams']['model_dims'][2]):
        #         self.L.append([subject, mri_path, slice_, 0])

        self.df = pd.DataFrame(self.L, columns=['Subject', 'Path MRI', 'Slice', 'Label'])
        self.df = self.df.assign(id=self.df.index.values).sample(frac=1)
        print(f'dataframe: \n{self.df["Label"].sum()} \n')


    def __len__(self):

        return self.df.shape[0]


    def __getitem__(self, index):

        load_slice = self.df.at[index, 'Slice']

        mri = preprocess(self.df.at[index, 'Path MRI'], self.config, norm=True)[:, :, load_slice]
        label = self.df.at[index, 'Label']

        return mri, label


def preprocess(path, config, norm=False):

    scan = nib.load(path)
    aff  = scan.affine
    vol  = scan.get_fdata() # np.int16(scan.get_fdata())
    
    new_affine = nibabel.affines.rescale_affine(aff, 
                                                vol.shape, 
                                                config['hyperparams']['new_z'], 
                                                config['hyperparams']['model_dims']
    ) 

    scan = nibabel.processing.conform(scan, 
                                      config['hyperparams']['model_dims'], 
                                      config['hyperparams']['new_z']
    )

    ni_img     = nib.Nifti1Image(scan.get_fdata(), new_affine)
    vol        = ni_img.get_fdata() 

    if norm:
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))

    return vol


