# Team Anonymous
# Auto Generation Model for Drawings

from keras.callbacks import LearningRateScheduler, TensorBoard
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np

import requests
import six
from six.moves import cStringIO as StringIO
import copy
import os
import sys
from keras.callbacks import Callback
import keras.backend as K



def batch_generator(dataset, train):
   
    count = 0  
    while True:
        if train:
            _, batch, s = dataset.random_batch()

        else:  
            count = 0 if count == dataset.num_batches else count
            _, batch, s = dataset.get_batch(count)
            count += 1

        encoder_input = batch[:, 1:dataset.max_seq_length + 1, :]
        expected_out = encoder_input

        decoder_input = batch[:, :dataset.max_seq_length, :]

        yield ({'encoder_input': encoder_input, 'decoder_input': decoder_input}, {'output': expected_out})

def load_dataset(data_dir, model_params)

    if isinstance(model_params.data_set, list):
        datasets = model_params.data_set
    else:
        datasets = [model_params.data_set]

    train_strokes = None
    valid_strokes = None
    test_strokes = None

    for dataset in datasets:
        data_filepath = os.path.join(data_dir, dataset)
        if data_dir.startswith('http://') or data_dir.startswith('https://'):
            print('Downloading %s', data_filepath)
            response = requests.get(data_filepath)
            data = np.load(StringIO(response.content))
        else:
            if six.PY3:
                data = np.load(data_filepath, encoding='latin1')
            else:
                data = np.load(data_filepath)
        print('Loaded {}/{}/{} from {}'.format( len(data['train']), len(data['valid']), len(data['test']), dataset))

        if train_strokes is None:
            train_strokes = data['train']
            valid_strokes = data['valid']
            test_strokes = data['test']
        else:
            train_strokes = np.concatenate((train_strokes, data['train']))
            valid_strokes = np.concatenate((valid_strokes, data['valid']))
            test_strokes = np.concatenate((test_strokes, data['test']))

    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    print('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
        len(all_strokes), len(train_strokes), len(valid_strokes),
        len(test_strokes), int(avg_len)))

    max_seq_len = get_max_len(all_strokes)
    model_params.max_seq_len = max_seq_len

    print('model_params.max_seq_len:', int(model_params.max_seq_len))

    train_set = DataLoader(
        train_strokes,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob)

    normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
    train_set.normalize(normalizing_scale_factor)

    valid_set = DataLoader(
        valid_strokes,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    valid_set.normalize(normalizing_scale_factor)

    test_set = DataLoader(
        test_strokes,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    test_set.normalize(normalizing_scale_factor)

    print('normalizing_scale_factor ', normalizing_scale_factor)

    output = [train_set, valid_set, test_set, model_params]
    return output


class Logger(object):
    def __init__(self, logsdir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(logsdir, 'log.txt'), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo):
        return DotDict([(copy.deepcopy(k, memo), copy.deepcopy(v, memo)) for k, v in self.items()])


class LearningRateSchedulerPerBatch(LearningRateScheduler):
    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerPerBatch, self).__init__(schedule, verbose)
        self.count = 0 

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_begin(self.count, logs)

    def on_batch_end(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_end(self.count, logs)
        self.count += 1


class KLWeightScheduler(Callback):

    def __init__(self, kl_weight, schedule, verbose=0):
        super(KLWeightScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.kl_weight = kl_weight
        self.count = 0  

    def on_batch_begin(self, batch, logs=None):

        new_kl_weight = self.schedule(self.count)
        if not isinstance(new_kl_weight, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')

        K.set_value(self.kl_weight, new_kl_weight)
        if self.verbose > 0 and self.count % 20 == 0:
            print('\nBatch %05d: KLWeightScheduler setting KL weight '
                  ' to %s.' % (self.count + 1, new_kl_weight))
        self.count += 1


class TensorBoardLR(TensorBoard):
    def __init__(self, *args, **kwargs):
        self.kl_weight = kwargs.pop('kl_weight')
        super().__init__(*args, **kwargs)
        self.count = 0

    def on_batch_end(self, batch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr),
                     'kl_weight': K.eval(self.kl_weight)})
        super().on_batch_end(batch, logs)


def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

def strokes_to_lines(strokes):
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines


def lines_to_strokes(lines):

    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]


def augment_strokes(strokes, prob=0.0):
    output = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            output.append(stroke)
    return np.array(output)


def scale_bound(stroke, average_dimension=10.0):
    bounds = get_bounds(stroke, 1)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    stroke[:, 0:2] /= (max_dimension / average_dimension)


def to_normal_strokes(big_stroke):
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    output = np.zeros((l, 3))
    output[:, 0:2] = big_stroke[0:l, 0:2]
    output[:, 2] = big_stroke[0:l, 3]
    return output


def clean_strokes(sample_strokes, factor=100):
    copy_stroke = []
    added_final = False
    for j in range(len(sample_strokes)):
        finish_flag = int(sample_strokes[j][4])
        if finish_flag == 0:
            copy_stroke.append([
                int(round(sample_strokes[j][0] * factor)),
                int(round(sample_strokes[j][1] * factor)),
                int(sample_strokes[j][2]),
                int(sample_strokes[j][3]), finish_flag
            ])
        else:
            copy_stroke.append([0, 0, 0, 0, 1])
            added_final = True
            break
    if not added_final:
        copy_stroke.append([0, 0, 0, 0, 1])
    return copy_stroke


def to_big_strokes(stroke, max_len=250):

    output = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    output[0:l, 0:2] = stroke[:, 0:2]
    output[0:l, 3] = stroke[:, 2]
    output[0:l, 2] = 1 - output[0:l, 3]
    output[l:, 4] = 1
    return output


def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


class DataLoader(object):

    def __init__(self,
                 strokes,
                 batch_size=100,
                 max_seq_length=250,
                 scale_factor=1.0,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 limit=1000):
        self.batch_size = batch_size  
        self.max_seq_length = max_seq_length  
        self.scale_factor = scale_factor  
        self.random_scale_factor = random_scale_factor 
        self.limit = limit
        self.augment_stroke_prob = augment_stroke_prob  
        self.start_stroke_token = [0, 0, 1, 0, 0] 
        self.preprocess(strokes)

    def preprocess(self, strokes):
        raw_data = []
        seq_len = []
        count_data = 0

        for i in range(len(strokes)):
            data = strokes[i]
            if len(data) <= (self.max_seq_length):
                count_data += 1
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data, dtype=np.float32)
                data[:, 0:2] /= self.scale_factor
                raw_data.append(data)
                seq_len.append(len(data))
        seq_len = np.array(seq_len)  
        idx = np.argsort(seq_len)
        self.strokes = []
        for i in range(len(seq_len)):
            self.strokes.append(raw_data[idx[i]])
        print("total images <= max_seq_len is %d" % count_data)
        self.num_batches = int(count_data / self.batch_size)

    def random_sample(self):
        sample = np.copy(random.choice(self.strokes))
        return sample

    def random_scale(self, data):
        x_scale_factor = (
            np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (
            np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        output = np.copy(data)
        output[:, 0] *= x_scale_factor
        output[:, 1] *= y_scale_factor
        return output

    def calculate_normalizing_scale_factor(self):
        data = []
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j, 0])
                data.append(self.strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, scale_factor=None):
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= self.scale_factor

    def batches_indices(self, indices):
        x_batch = []
        seq_len = []
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.random_scale(self.strokes[i])
            data_copy = np.copy(data)
            if self.augment_stroke_prob > 0:
                data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            x_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
        seq_len = np.array(seq_len, dtype=int)
        return x_batch, self.con_batch(x_batch, self.max_seq_length), seq_len

    def random_batch(self):
        idx = np.random.permutation(range(0, len(self.strokes)))[0:self.batch_size]
        return self.batches_indices(idx)

    def get_batch(self, idx):
        assert idx >= 0, "idx must be non negative"
        assert idx < self.num_batches, "idx must be less than the number of batches:"
        start_idx = idx * self.batch_size
        indices = range(start_idx, start_idx + self.batch_size)
        return self.batches_indices(indices)

    def con_batch(self, batch, max_len):
        output = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        assert len(batch) == self.batch_size
        for i in range(self.batch_size):
            l = len(batch[i])
            assert l <= max_len
            output[i, 0:l, 0:2] = batch[i][:, 0:2]
            output[i, 0:l, 3] = batch[i][:, 2]
            output[i, 0:l, 2] = 1 - output[i, 0:l, 3]
            output[i, l:, 4] = 1
            output[i, 1:, :] = output[i, :-1, :]
            output[i, 0, :] = 0
            output[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
            output[i, 0, 3] = self.start_stroke_token[3]
            output[i, 0, 4] = self.start_stroke_token[4]
        return output