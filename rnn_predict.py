import numpy as np
import random
import math
import h5py
from scipy.io import savemat
import theano
from theano import tensor as T
import lasagne
import lasagne.layers as layers
from lasagne.nonlinearities import softmax, sigmoid, rectify

class import_model(object):
    def __init__(self, param, dim_point=3, num_joints=25, num_class=120):
        self._param = param
        self._dim_point = dim_point
        self._num_joints = num_joints
        self._num_class = num_class


    def smooth_skeleton(self, skeleton):
        assert(skeleton.shape[2] == 3), ' input must be skeleton array'
        filt = np.array([-3,12,17,12,-3])/35.0
        skt = np.concatenate((skeleton[0:2], skeleton, skeleton[-2:]), axis=0)
        for idx in xrange(2, skt.shape[0]-2):
            skeleton[idx-2] = np.swapaxes(np.dot(np.swapaxes(skt[idx-2:idx+3], 0, -1), filt), 0, -1)
        return skeleton


    def calculate_height(self, skeleton):
        center1 = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
        w1 = skeleton[:,23,:] - center1
        w2 = skeleton[:,22,:] - center1
        center2 = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4
        h0 = skeleton[:,3,:] - center2
        h1 = skeleton[:,19,:] - center2
        h2 = skeleton[:,15,:] - center2
        width = np.max([np.max(np.abs(w1[:,0])), np.max(np.abs(w2[:,0]))])
        heigh = np.max([np.max(np.abs(h1[:,1])), np.max(np.abs(h2[:,1])), np.max(h0[:,1])])
        return width, heigh


    def subtract_mean(self, skeleton, smooth=False, scale=True):
        if smooth:
            skeleton = self.smooth_skeleton(skeleton)
        # substract mean values
        # notice: use two different mean values to normalize skeleton data
        if 0:
            center = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
            # center = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4
            for idx in xrange(skeleton.shape[1]):
                skeleton[:, idx] = skeleton[:, idx] - center
        center1 = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
        center2 = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4

        for idx in [24,25,12,11,10,9, 5,6,7,8,23,22]:
            skeleton[:, idx-1] = skeleton[:, idx-1] - center1
        for idx in (set(range(1, 1+skeleton.shape[1]))-set([24,25,12,11,10,9,  5,6,7,8,23,22])):
            skeleton[:, idx-1] = skeleton[:, idx-1] - center2

        if scale:
            width, heigh = self.calculate_height(skeleton)
            scale_factor1, scale_factor2 = 0.36026082, 0.61363413
            skeleton[:,:,0] = scale_factor1*skeleton[:,:,0]/width
            skeleton[:,:,1] = scale_factor2*skeleton[:,:,1]/heigh
        return skeleton


    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=True):
        assert len(inputs) == len(targets)
        indices = np.arange(len(inputs))
        if shuffle:
            np.random.shuffle(indices)
        if 0:
            for start_idx in range(0, len(inputs) - len(inputs) % batchsize, batchsize):
                excerpt = indices[start_idx:start_idx + batchsize]
                y = [targets[s] for s in excerpt]
                x = np.asarray([inputs[s] for s in excerpt])
                yield x, y
        start_idx = 0
        num_batch = (len(inputs) + batchsize - 1) / batchsize
        for batch_idx in range(0, num_batch):
            if batch_idx == num_batch-1:
                excerpt = indices[start_idx:]
            else:
                excerpt = indices[start_idx:start_idx + batchsize]
            start_idx = start_idx + batchsize
            y = [targets[s] for s in excerpt]
            x = np.asarray([inputs[s] for s in excerpt])
            yield x, y


    def load_sample_step_list(self, h5_file, list_file, num_seq, step=1, start_zero=True, sub_mean=False, scale=False, smooth=False):
        name_list = [line.strip() for line in open(list_file, 'r').readlines()]
        label_list = [(int(name[17:20])-1) for name in name_list]
        X = []
        label = []
        with h5py.File(h5_file,'r') as hf:
            for idx, name in enumerate(name_list):
                skeleton = np.asarray(hf.get(name))
                if sub_mean:
                    skeleton = self.subtract_mean(skeleton, smooth=smooth, scale=scale)
                for start in range(0, 1 if start_zero else step):
                    skt = skeleton[start:skeleton.shape[0]:step]
                    if skt.shape[0] > num_seq:
                        # process sequences longer than num_seq, sample two sequences, if start_zero=True, only sample once from 0
                        for sidx in ([np.arange(num_seq)] if start_zero else [np.arange(num_seq), np.arange(skt.shape[0]-num_seq, skt.shape[0])]):
                            X.append(skt[sidx])
                            label.append(label_list[idx])
                    else:
                        if skt.shape[0] < 0.05*num_seq: # skip very small sequences
                            continue
                        skt = np.concatenate((np.zeros((num_seq-skt.shape[0], skt.shape[1], skt.shape[2])), skt), axis=0)
                        X.append(skt)
                        label.append(label_list[idx])

        X = np.asarray(X)
        label = (np.asarray(label)).astype(np.int32)
        # rearrange skeleton data by part
        X = self.divide_skeleton_part(X)
        # X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))
        X = X.astype(np.float32)
        return X, label


    def predict()
        input_var = T.tensor3('X')
        network = lasagne.layers.InputLayer((None, None, self._dim_point*self._num_joints), input_var)
        network = bi_direction_lstm(network, 512, only_return_final=False)
        network = bi_direction_lstm(network, 512, only_return_final=False)
        network = lasagne.layers.ExpressionLayer(network, lambda X: X.max(1), output_shape='auto')
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5), self._num_class, nonlinearity=softmax)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        predict_fn = theano.function([input_var], test_prediction)
        valX, valY = self.load_sample_step_list(self._param['tst_arr_file'], self._param['tst_lst_file'], self._param['num_seq'],
                step=self._param['step'], start_zero=True, scale=self._param['scale'],
                sub_mean=self._param['sub_mean'], smooth=self._param['smooth'])

        with h5py.File(self._param['initial_file'],'r') as f:
            param_values = []
            for i in range(0, len(f.keys())):
                param_values.append( np.asarray(f.get('arr_%d' % i)) )
            lasagne.layers.set_all_param_values(network, param_values)
            val_predictions = np.zeros((0, self._num_class))
            for x, y in self.iterate_minibatches(valX, valY, self._param['batchsize'], shuffle=False):
                val_predictions = np.concatenate((val_predictions, predict_fn(x)), axis=0)
            save_name='data/result.mat'
            savemat(save_name, mdict={'predictions':val_predictions, 'gt_label':valY} )


def run_model():
    param = {}

    param['tst_arr_file'] = 'data/subj_seq/array_list_train.h5'
    param['tst_lst_file'] = 'data/subj_seq/file_list_train.txt'
    param['initial_file'] = 'data/link_to_model'

    param['num_seq'] = 100
    param['step'] = 1
    param['rand_start'] = True
    param['max_start_rate'] = 0.3 # 0.3*length
    param['rand_view'] = False
    param['sub_mean']=True
    param['scale']=False
    param['smooth']=False

    model = import_model(param)
    model.predict()

if __name__ == '__main__':
    run_model()
