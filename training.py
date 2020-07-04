from __future__ import print_function
import os
import numpy as np
import pandas as pd
import imp
import datetime
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    pass    
import glob
import sys
import tqdm
import argparse
import pathlib
import tensorflow as tf
from tensorflow.keras import layers, models
from interaction import InteractionModel

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
if os.path.isdir('/storage/group/gpu/bigdata/BumbleB'):
    test_path = 'storage/group/gpu/bigdata/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
    train_path = '/storage/group/gpu/bigdata/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_train_val/'
elif os.path.isdir('/eos/user/w/woodson/IN'):
    test_path = '/eos/user/w/woodson/IN/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
    train_path = '/eos/user/w/woodson/IN/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_train_val/'

N = 60 # number of charged particles
N_sv = 5 # number of SVs 
n_targets = 2 # number of classes

params_2 = ['track_ptrel',     
          'track_erel',     
          'track_phirel',     
          'track_etarel',     
          'track_deltaR',
          'track_drminsv',     
          'track_drsubjet1',     
          'track_drsubjet2',
          'track_dz',     
          'track_dzsig',     
          'track_dxy',     
          'track_dxysig',     
          'track_normchi2',     
          'track_quality',     
          'track_dptdpt',     
          'track_detadeta',     
          'track_dphidphi',     
          'track_dxydxy',     
          'track_dzdz',     
          'track_dxydz',     
          'track_dphidxy',     
          'track_dlambdadz',     
          'trackBTag_EtaRel',     
          'trackBTag_PtRatio',     
          'trackBTag_PParRatio',     
          'trackBTag_Sip2dVal',     
          'trackBTag_Sip2dSig',     
          'trackBTag_Sip3dVal',     
          'trackBTag_Sip3dSig',     
          'trackBTag_JetDistVal'
         ]

params_3 = ['sv_ptrel',
          'sv_erel',
          'sv_phirel',
          'sv_etarel',
          'sv_deltaR',
          'sv_pt',
          'sv_mass',
          'sv_ntracks',
          'sv_normchi2',
          'sv_dxy',
          'sv_dxysig',
          'sv_d3d',
          'sv_d3dsig',
          'sv_costhetasvpv'
         ]

def main(args):
    """ Main entry point of the app """
    
    #Convert two sets into two branch with one set in both and one set in only one (Use for this file)
    params = params_2
    params_sv = params_3
    
    from data import H5Data
    files = glob.glob(train_path + "/newdata_*.h5")
    files_val = files[4:5] # take first 5 for validation
    files_train = files[5:6] # take rest for training
    
    label = 'new'
    outdir = args.outdir
    vv_branch = args.vv_branch
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)  

    batch_size = 256
    data_train = H5Data(batch_size = batch_size,
                        cache = None,
                        preloading=0,
                        features_name='training_subgroup', 
                        labels_name='target_subgroup',
                        spectators_name='spectator_subgroup')
    data_train.set_file_names(files_train)
    data_val = H5Data(batch_size = batch_size,
                      cache = None,
                      preloading=0,
                      features_name='training_subgroup', 
                      labels_name='target_subgroup',
                      spectators_name='spectator_subgroup')
    data_val.set_file_names(files_val)

    n_val=data_val.count_data()
    n_train=data_train.count_data()

    print("val data:", n_val)
    print("train data:", n_train)

    net_args = (N, n_targets, len(params), args.hidden, N_sv, len(params_sv))
    net_kwargs = {"vv_branch": int(vv_branch), "De": args.De, "Do": args.Do}
    
    gnn = InteractionModel(*net_args, **net_kwargs)
    gnn.compile(optimizer='adam')
    print("Model compiled")
    #### Start training ####
    
    n_epochs = 1
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    val_loss_results = []
    val_accuracy_results = []
    
    # Log directory for Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    pathlib.Path(train_log_dir).mkdir(parents=True, exist_ok=True)  
    pathlib.Path(test_log_dir).mkdir(parents=True, exist_ok=True)  

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(n_epochs):
        
        # Tool to keep track of the metrics
        epoch_loss_avg = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
        val_epoch_loss_avg = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        val_epoch_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

        # Training
        for sub_X,sub_Y,sub_Z in tqdm.tqdm(data_train.generate_data(),total = n_train/batch_size):
            training = sub_X[2]
            training_sv = sub_X[3]
            target = sub_Y[0]

            # Define loss function
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            def loss(model, x1, x2, y):
                y_ = model([x1, x2])
                return cce(y_true=y, y_pred=y_)
            def grad(model, input_par, input_sv, targets):
                with tf.GradientTape() as tape:
                    loss_value = loss(model, input_par, input_sv, targets)
                return loss_value, tape.gradient(loss_value, model.trainable_variables)
            
            # Define optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            # Compute loss and gradients
            loss_value, grads = grad(gnn, training, training_sv, target)

            # Update the gradients
            optimizer.apply_gradients(zip(grads, gnn.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            epoch_accuracy(target, tf.nn.softmax(gnn([training, training_sv])))

        # Validation
        for sub_X,sub_Y,sub_Z in tqdm.tqdm(data_val.generate_data(),total = n_val/batch_size):
            training = sub_X[2]
            training_sv = sub_X[3]
            target = sub_Y[0]
            
            # Compute the loss
            loss_value = loss(gnn, training, training_sv, target)
            
            # Track progress
            val_epoch_loss_avg(loss_value)
            val_epoch_accuracy(target, tf.nn.softmax(gnn([training, training_sv])))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        val_loss_results.append(val_epoch_loss_avg.result())
        val_accuracy_results.append(val_epoch_accuracy.result())
        
        # Logs for tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', epoch_accuracy.result(), step=epoch)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', val_epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', val_epoch_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(epoch+1,
                         epoch_loss_avg.result(), 
                         epoch_accuracy.result()*100,
                         val_epoch_loss_avg.result(), 
                         val_epoch_accuracy.result()*100))

        # Reset metrics every epoch
        epoch_loss_avg.reset_states()
        val_epoch_loss_avg.reset_states()
        epoch_accuracy.reset_states()
        val_epoch_accuracy.reset_states()

    # Save the model after training
    save_path = 'models/2/'
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)  
    tf.saved_model.save(gnn, save_path)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    parser.add_argument("vv_branch", help="Required positional argument")
    
    # Optional arguments
    parser.add_argument("--De", type=int, action='store', dest='De', default = 5, help="De")
    parser.add_argument("--Do", type=int, action='store', dest='Do', default = 6, help="Do")
    parser.add_argument("--hidden", type=int, action='store', dest='hidden', default = 15, help="hidden")

    args = parser.parse_args()
    main(args)
