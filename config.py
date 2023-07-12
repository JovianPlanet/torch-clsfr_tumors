import os
import datetime

def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    hyperparams = {'model_dims': (128, 128, 64),
                   'lr'        : 0.0001,
                   'epochs'    : 20, #20
                   'batch_size': 4,
                   'new_z'     : [2, 2, 2],
                   'n_heads'   : 67,
                   'n_train'   : 100, #295 #Total=295
                   'n_val'     : 10, #37
                   'n_test'    : 10, #37
                   'batchnorm' : False,
                   'nclasses'  : 1,
                   'thres'     : 0.5
    }

    labels = {'bgnd': 0, # Image background
              'NCR' : 1, # necrotic tumor core
              'ED'  : 2, # peritumoral edematous/invaded tissue
              'ET'  : 4, # GD-enhancing tumor
    }

    brats_train = os.path.join('/home',
                               'davidjm',
                               'Downloads',
                               'BraTS-dataset',
                               'train',
                               #'MICCAI_BraTS2020_TrainingData'
    )

    brats_val = os.path.join('/home',
                             'davidjm',
                             'Downloads',
                             'BraTS-dataset',
                             'val',
    )

    brats_test = os.path.join('/home',
                             'davidjm',
                             'Downloads',
                             'BraTS-dataset',
                             'test',
    )

    nfbs_train = os.path.join('/home',
                           'davidjm',
                           'Downloads',
                           'Reg-NFBS',
                           'train'
    )

    nfbs_val = os.path.join('/home',
                           'davidjm',
                           'Downloads',
                           'Reg-NFBS',
                           'val'
    )

    nfbs_test = os.path.join('/home',
                           'davidjm',
                           'Downloads',
                           'Reg-NFBS',
                           'test'
    )

    datasets = {'brats_train': brats_train, 'brats_val': brats_val, 'brats_test': brats_test,
                'nfbs_train': nfbs_train, 'nfbs_val': nfbs_val, 'nfbs_test': nfbs_test}


    if mode == 'train':

        files = {
        'model': 'weights-BCELog-'+str(hyperparams['epochs'])+'_eps-'+str(hyperparams['n_train'])+'_heads-'+str(datetime.date.today())+'-_nobn',
        'losses': './outs/losses-BCELog-'+str(hyperparams['epochs'])+'_eps-'+str(hyperparams['n_train'])+'_heads-'+str(datetime.date.today())+'-_nobn.csv',
        't_accus': './outs/t-accs-BCELog-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'-_nobn.csv',
        'v_accus': './outs/v-accs-BCELog-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'-_nobn.csv',
        'pics': './outs/imgs/BCELog-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today()),
        }

        res_path = os.path.join('/media',
                                'davidjm',
                                'Disco_Compartido',
                                'david',
                                'torch-clsfr_tumors',
                                'results'
        )

        return {'mode': mode,
                'data': datasets,
                'hyperparams': hyperparams,
                'files': files,
                'res_path': res_path, 
                'labels': labels,
        }

    elif mode == 'test':

        PATH_TRAINED_MODEL = 'prueba.pth' # 'weights-bcedice-20_eps-100_heads-2023-03-10-_nobn.pth'
        PATH_TEST_DICES = './outs/dice_coeff'+PATH_TRAINED_MODEL[7:-4]+'-test.csv'

        return {'mode'      : mode,
                'data'      : datasets,
                'hyperparams': hyperparams,
                'thres'     : threshold,
                'labels'    : labels,
                'weights'   : PATH_TRAINED_MODEL,
                'test_fn'   : PATH_TEST_DICES,
        }

    elif mode == 'assess':

        train_losses = './outs/losses-bcedice-20_eps-100_heads-2023-03-10-_nobn.csv'
        train_dices  = './outs/dices-bcedice-20_eps-100_heads-2023-03-10-_nobn.csv'
        test_dices = './outs/dice_coeff-bcedice-20_eps-100_heads-2023-03-10-_nobn-test.csv'

        files = {'train_Loss': train_losses,
                 'train_Dice': train_dices,
                 'test_Dice' : test_dices}

        return {'mode'     : mode,
                'labels'   : labels,
                'losses_fn': losses_fn,
                'dices_fn' : dices_fn,
                'files'    : files}
