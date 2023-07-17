import os
from datetime import datetime
from pathlib import Path


def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    hyperparams = {'model_dims': (128, 128, 64), # Dimensiones de entrada al modelo
                   'new_z'     : [2, 2, 2],      # Nuevo tama;o de zooms
                   'lr'        : 0.0001,         # Taza de aprendizaje
                   'epochs'    : 20,             # Numero de epocas
                   'batch_size': 4,              # Tama;o del batch
                   'crit'      : 'BCELog',       # Fn de costo. Opciones: 'CELoss', 'BCELog'
                   'n_train'   : 100,            # "" Entrenamiento
                   'n_val'     : 10,             # "" Validacion
                   'n_test'    : 10,             # "" Prueba
                   'batchnorm' : False,          # Normalizacion de batch
                   'nclasses'  : 1,              # Numero de clases
                   'thres'     : 0.5,            # Umbral
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

    folder = './outs/Ex-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    Path(folder).mkdir(parents=True, exist_ok=True)
    #Path(os.path.join(folder, 'imgs')).mkdir(parents=True, exist_ok=True)

    if mode == 'train':

        files = {'model'  : os.path.join(folder, 'weights'),
                 'losses' : os.path.join(folder, 'losses.csv'),
                 't_mets' : os.path.join(folder, 'train_metrics.csv'),
                 'v_mets' : os.path.join(folder, 'val_metrics.csv'),
                 'params' : os.path.join(folder, 'params.txt'),
                 'summary': os.path.join(folder, 'cnn_summary.txt'),
                 'pics'   : os.path.join(folder, 'imgs'),
                }


        return {'mode'       : mode,
                'data'       : datasets,
                'hyperparams': hyperparams,
                'files'      : files,
                'labels'     : labels,
        }

    elif mode == 'test':

        PATH_TRAINED_MODEL = './outs/2023-07-11/weights-BCELog-20_eps-100_heads-2023-07-11-_nobn-e19.pth' 
        PATH_TEST_DICES = './outs/2023-07-11/test_metrics.csv'

        return {'mode'       : mode,
                'data'       : datasets,
                'hyperparams': hyperparams,
                'labels'     : labels,
                'weights'    : PATH_TRAINED_MODEL,
                'test_fn'    : PATH_TEST_DICES,
        }

    elif mode == 'assess':

        train_losses = './outs/2023-07-11/losses-BCELog-20_eps-100_heads-2023-07-11-_nobn.csv'
        train_metrics  = './outs/2023-07-11/t-accs-BCELog-20_eps-2023-07-11-_nobn.csv'
        val_metrics  = './outs/2023-07-11/v-accs-BCELog-20_eps-2023-07-11-_nobn.csv'
        test_metrics   = './outs/2023-07-11/test_metrics.csv'

        files = {'train_Loss': train_losses,
                 'train_mets': train_metrics,
                 'val_mets'  : val_metrics,
                 'test_mets' : test_metrics
                }

        return {'mode'     : mode,
                'labels'   : labels,
                'files'    : files
                }
