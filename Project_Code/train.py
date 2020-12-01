# Team Anonymous
# Auto Generation Model for Drawings

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

from model import rnnBasedVAE, get_default_hparams
from keras.callbacks import ModelCheckpoint
from vae_utils import load_dataset,batch_generator, KLWeightScheduler, LearningRateSchedulerPerBatch, TensorBoardLR, DotDict, Logger


def get_out_dict(rnn_based_VAE, model_params, experiment_path=''):
    out_dict = {}

    out_dict['model_checkpoint'] = ModelCheckpoint(filepath=os.path.join(experiment_path, 'checkpoints','weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', save_best_only=True, mode='min')
    out_dict['lr_schedule'] = LearningRateSchedulerPerBatch(
        lambda step: ((model_params.learning_rate - model_params.min_learning_rate) * model_params.decay_rate ** step
                      + model_params.min_learning_rate))

    out_dict['kl_weight_schedule'] = KLWeightScheduler(schedule=lambda step:
                                       (model_params.kl_weight - (model_params.kl_weight - model_params.kl_weight_start)
                                       * model_params.kl_decay_rate ** step), kl_weight=rnn_based_VAE.kl_weight, verbose=1)

    # Tensorboard 
    out_dict['tensorboard'] = TensorBoardLR(log_dir=os.path.join(experiment_path, 'tensorboard'),
                                    kl_weight=rnn_based_VAE.kl_weight, update_freq=model_params.batch_size*25)

    return out_dict


def main(args, hyper_parameters):

    logsdir = os.path.join(args.experiment_dir, 'logs')
    os.makedirs(logsdir)
    os.makedirs(os.path.join(args.experiment_dir, 'checkpoints'))
    sys.stdout = Logger(logsdir)

    hparams_dot = DotDict(hyper_parameters)

    hparams_dot.data_set = args.data_set
    datasets = load_dataset(args.data_dir, hparams_dot)

    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]
    model_params = datasets[3]

    rnn_based_VAE = rnnBasedVAE(model_params)
    rnn_based_VAE.compile()
    model = rnn_based_VAE.model

    train_generator = batch_generator(train_set, train=True)
    val_generator = batch_generator(valid_set, train=False)

    model_callbacks = get_out_dict(rnn_based_VAE=rnn_based_VAE, model_params=model_params, experiment_path=args.experiment_dir)

    if args.checkpoint is not None:
        rnn_based_VAE.load_trained_weights(args.checkpoint)
        num_batches = model_params.save_every if model_params.save_every is not None else train_set.num_batches
        count = args.initial_epoch*num_batches
        model_callbacks['lr_schedule'].count = count
        model_callbacks['kl_weight_schedule'].count = count

    with open(os.path.join(logsdir, 'model_config.json'), 'w') as f:
        json.dump(model_params, f, indent=True)

    steps_per_epoch = model_params.save_every if model_params.save_every is not None else train_set.num_batches
    model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=model_params.epochs,
                        validation_data=val_generator, validation_steps=valid_set.num_batches,
                        callbacks=[cbk for cbk in model_callbacks.values()],
                        initial_epoch=args.initial_epoch)


if __name__ == '__main__':

    hyper_parameters = get_default_hparams()

    parser = argparse.ArgumentParser(description='Run RNN Based VAE')

    parser.add_argument('--data_dir', type=str,
                        default='datasets',
                        help='Data path. (default: %(default)s)')
    parser.add_argument('--data_set', type=str,
                        default=hyper_parameters['data_set'],
                        help='Class of .npz file. (default: %(default)s)')

    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help='Path of checkpoint')

    parser.add_argument('--initial_epoch', type=int,
                        default=0,
                        help='Load from checkpoint. (default: %(default)s)')

    args = parser.parse_args()

    if isinstance(args.data_set, list): 
        sets = [os.path.splitext(s)[0] for s in args.data_set]
        experiment_path = os.path.join(args.experiment_dir, "{}\\exp".format('_'.join(sets)))
        args.data_set = [s+'.npz' for s in sets]
    else:
        data_set = os.path.splitext(args.data_set)[0]
        experiment_path = os.path.join(args.experiment_dir, "{}\\exp".format(data_set))
        args.data_set = data_set+'.npz'

    dir_counter = 0
    new_experiment_path = experiment_path
    while os.path.exists(new_experiment_path):
        new_experiment_path = experiment_path + '_' + str(dir_counter)
        dir_counter += 1

    args.experiment_dir = new_experiment_path
    main(args, hyper_parameters)