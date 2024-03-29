#!/miniconda/bin/python3.8 

'''
Cat Updates
- Added path to create weights folder
- created two separate trainers (one for cpu and one for gpu)
- defined cpu when gpu is = -1
- switched order of imports for comet_ml and torch 

'''


import os
import yaml
import numpy as np
import typer
from munch import Munch

import comet_ml ## python says it wants comet before torch
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, CometLogger, TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
#import torch ## cat edit, they want this loaded after 

from src import algorithms
from src import datasets
import IPython

app = typer.Typer()#pretty_exceptions_show_locals=False)

# experiment = Experiment(
#     api_key="crD0w5pAk59gNUJ88yfNuMo5F",
#     project_name="snowpole-segmentation",
#     workspace="catherine-m-breen",
# )

@app.command()
def main(
         config:str='./configs/snowpole_plain_033123.yaml',
         project:str='SNOWPOLES_TEST',  
         gpus:str='0', 
         logger_type:str='tensorboard',
         evaluate:str=None,
         np_threads:str='8',
         session:int=0,
         seed:int=0,
         dev:bool=False
         ):

    ############
    # Set gpus #
    ############
    #IPython.embed()
    gpus = gpus #if torch.cuda.is_available() else 'cpu'
    gpus = [int(i) for i in gpus.split(',')] # 'cpu' # cat edit: for gpu
  
    #############################
    # Set environment variables #
    #############################
    # Set numpy threads
    os.environ["OMP_NUM_THREADS"] = str(np_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(np_threads)
    os.environ["MKL_NUM_THREADS"] = str(np_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(np_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(np_threads)
    #os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = 'gloo'
    

    #######################
    # Load configurations #
    #######################
    with open(config) as f:
        conf = Munch(yaml.load(f, Loader=yaml.FullLoader))

    if gpus != [-1]:
        if len(gpus) >= 1:
            conf.batch_size = int(conf.batch_size / torch.cuda.device_count()) # could also try 2
    else: conf.batch_size = conf.batch_size

    pl.seed_everything(seed)

    ###########################
    # Load data and algorithm #
    ###########################
    dataset = datasets.__dict__[conf.dataset_name](conf=conf)
    learner = algorithms.__dict__[conf.algorithm](conf=conf)

    print('training size:', dataset.dset_tr.__len__())
    print('testing size:', dataset.dset_te.__len__())
    
    ###############
    # Load logger #
    ###############
    log_folder = 'log_dev' if dev else 'log'
    if logger_type == 'csv':
        logger = CSVLogger(
            save_dir='./{}/{}'.format(log_folder, conf.algorithm),
            prefix=project,
            name='{}_{}_{}'.format(conf.algorithm, conf.conf_id, session),
            version=session
        )
    elif logger_type == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir='./{}/{}'.format(log_folder, conf.algorithm),
            prefix=project,
            name='{}_{}_{}'.format(conf.algorithm, conf.conf_id, session),
            version=session
        )
    elif logger_type == 'comet':
        logger = CometLogger(
            api_key= conf.comet_api, 
            save_dir='./{}/{}'.format(log_folder, conf.algorithm),
            project_name=project,  # Optional
            experiment_name='{}_{}_{}'.format(conf.algorithm, conf.conf_id, session),
        )

    ##################
    # Load callbacks #
    ##################
    weights_folder = 'weights_dev' if dev else 'weights'
    if not os.path.exists('./{}/{}'.format(weights_folder, conf.algorithm)):  
        os.makedirs('./{}/{}'.format(weights_folder, conf.algorithm), exist_ok=True)

    checkpoint_callback = ModelCheckpoint(monitor='valid_mean_IoU', mode='max', dirpath='./{}/{}'.format(weights_folder, conf.algorithm),
        filename='{}-{}'.format(conf.conf_id, session) + '-{epoch:02d}-{valid_mean_IoU:.2f}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    #IPython.embed()
    #################
    # Setup trainer #
    #################
    if gpus == [-1]: ## notation for cpu
        trainer = pl.Trainer(
            max_steps=conf.num_iters,
            check_val_every_n_epoch=1, 
            log_every_n_steps = conf.log_interval, 
            accelerator = 'cpu', # cat edit: comment if using gpu
            #gpus=gpus, # cat edit: comment out if no gpu
            logger=None if evaluate is not None else logger,
            callbacks=[lr_monitor, checkpoint_callback],
            # strategy=DDPStrategy(find_unused_parameters=True) if len(gpus) > 1 else 'dp', # cat edit: comment out if no gpu
            num_sanity_val_steps=0, ## could make it 2 if you want to check your validation steps
            profiler='simple',
            enable_progress_bar=True,
        )
    else: 
        #IPython.embed() ## make sure you have the right pytorch version 
    #     trainer = pl.Trainer(
    #         max_steps=conf.num_iters,
    #         check_val_every_n_epoch=1, 
    #         log_every_n_steps = conf.log_interval, 
    #         #gpus=gpus,
    #         logger=None if evaluate is not None else logger,
    #         callbacks=[lr_monitor, checkpoint_callback],
    #         #devices=gpus,
    #         accelerator='auto',
    #         #strategy=DDPStrategy(find_unused_parameters=True) if len(gpus) >= 1 else 'dp',
    #         num_sanity_val_steps=0,
    #         profiler='simple',
    #         enable_progress_bar=True,
    #         fast_dev_run=True,
    # )
        trainer = pl.Trainer(
            max_steps=conf.num_iters,
            check_val_every_n_epoch=2, 
            log_every_n_steps = conf.log_interval, 
            logger=None if evaluate is not None else logger,
            callbacks=[lr_monitor, checkpoint_callback],
            devices=1,
            accelerator='gpu',
            strategy=DDPStrategy(find_unused_parameters=True) if len(gpus) >= 1 else 'dp',
            num_sanity_val_steps=0,
            profiler='simple',
            enable_progress_bar=True,
            #fast_dev_run=True,
    )


    #######
    # RUN #
    #######
    if evaluate is not None:
        trainer.validate(learner, datamodule=dataset, ckpt_path=evaluate)
    else:
        #IPython.embed()e
        # print(dataset.dset_tr.__getitem__(4)[1].device)
        # print(dataset.dset_tr.__getitem__(4)[1].dtype)
        # print(dataset.dset_tr.__getitem__(4)[1].shape)
        # print('device of model', next(learner.parameters()).device)
        # print('device of model', next(learner.parameters()).dtype)
        trainer.fit(learner, datamodule=dataset) #train_dataloader=dataset.dset_tr, val_dataloader=dataset.dset_te) #datamodule=dataset)
        #trainer.fit(learner, dataset.dset_tr, dataset.dset_te) #datamodule=dataset))

if __name__ == '__main__':
    app()