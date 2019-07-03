import os
import numpy as np
import random
import h5py
# import cv2

class ntu_rgbd(object):
    def __init__(self, data_path):
        self._data_path = data_path

    def skeleton_miss_list(self):
        lines = open('data/samples_with_missing_skeletons.txt', 'r').readlines()
        return [line.strip()+'.skeleton' for line in lines]

    def get_multi_subject_list(self):
        lines = open('data/samples_with_multi_subjects.txt', 'r').readlines()
        return [line.strip() for line in lines]

    def smooth_skeleton(self, skeleton):
        assert(skeleton.shape[2] == 3), ' input must be skeleton array'
        filt = np.array([-3,12,17,12,-3])/35.0
        skt = np.concatenate((skeleton[0:2], skeleton, skeleton[-2:]), axis=0)
        for idx in xrange(2, skt.shape[0]-2):
            skeleton[idx-2] = np.swapaxes(np.dot(np.swapaxes(skt[idx-2:idx+3], 0, -1), filt), 0, -1)
        return skeleton

    def subtract_mean(skeleton, smooth=False):
        if smooth:
            skeleton = self.smooth_skeleton(skeleton)
        # substract mean values
        center = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
        for idx in xrange(skeleton.shape[1]):
            skeleton[:, idx] = skeleton[:, idx] - center
        return skeleton

    def load_skeleton_file_multi_subject(self, filename, sub_idx=1, num_joints=25):
        # sub_idx, subject index, 1, 2
        # return ndarray, n_step*n_joint*7 (3 postion, 4 angle)
        lines = open(os.path.join(self._data_path, filename), 'r').readlines()
        # notice: determine the number of step, not sure which is better
        step = int(lines[0].strip())
        skeleton = np.zeros((step, num_joints, 7))
        start = 1
        sidx = [0,1,2,7,8,9,10]
        idx = 0
        while start < len(lines): # and idx < step
            if sub_idx==1:
                if lines[start].strip() in ['1', '2', '3', '4']:
                    skeleton[idx] = np.asarray([map(float, np.array(line_per.strip().split())[sidx]) \
                                                for line_per in lines[start+1+2:start+26+2]])
                    idx = idx + 1
                    start = start + 26 + 2
                else:
                    start = start + 1
            if sub_idx==2:
                if lines[start].strip() in ['2', '3', '4']:
                    skeleton[idx] = np.asarray([map(float, np.array(line_per.strip().split())[sidx]) \
                                                for line_per in lines[start+1+2+27:start+26+2+27]])
                    idx = idx + 1
                    start = start + 1 + 26 + 2 + 27
                else:
                    start = start + 1
            if sub_idx==3:
                if lines[start].strip() in ['3', '4']:
                    skeleton[idx] = np.asarray([map(float, np.array(line_per.strip().split())[sidx]) \
                                                for line_per in lines[start+1+2+27+27:start+26+2+27+27]])
                    idx = idx + 1
                    start = start + 1 + 26 + 2 + 27 + 27
                else:
                    start = start + 1
        return skeleton[0:idx]

    def save_h5_file_skeleton_list(self, save_home, trn_list, split='train', angle=False):
        if 0:
            multi_list = self.get_multi_subject_list()
            one_list = list(set(trn_list) - set(multi_list))
            multi_list = list(set(trn_list) - set(one_list))

        # save file list to txt
        save_name = os.path.join(save_home, 'file_list_' +  split + '.txt')
        with open(save_name, 'w') as fid_txt:  # fid.write(file+'\n')
            # save array list to hdf5
            save_name = os.path.join(save_home, 'array_list_' + split + '.h5')
            with h5py.File(save_name, 'w') as fid_h5:
                for fn in trn_list:
                    skeleton_set, pid_set, std_set = self.person_position_std(fn)
                    # filter skeleton by standard value
                    count = 0
                    for idx2 in xrange(len(pid_set)):
                        if std_set[idx2][0] > 0.1 or std_set[idx2][1] > 0.1:
                            count = count + 1
                            name=fn+pid_set[idx2]
                            if angle:
                                fid_h5[name] = skeleton_set[idx2][:,:, 3:]
                            else:
                                fid_h5[name] = skeleton_set[idx2][:,:, 0:3]
                            fid_txt.write(name + '\n')
                    if count == 0:
                        std_sum = [np.sum(it) for it in std_set]
                        idx2 = np.argmax(std_sum)
                        name=fn+pid_set[idx2]
                        if angle:
                            fid_h5[name] = skeleton_set[idx2][:,:, 3:]
                        else:
                            fid_h5[name] = skeleton_set[idx2][:,:, 0:3]
                        fid_txt.write(name + '\n')

    def person_position_std(self, filename, num_joints=25):
        lines = open(os.path.join(self._data_path, filename), 'r').readlines()
        step = int(lines[0].strip())
        pid_set = []
        # idx_set length of sequence
        idx_set = []
        skeleton_set = []
        start = 1
        sidx = [0,1,2,7,8,9,10]
        while start < len(lines): # and idx < step
            if lines[start].strip()=='25':
                pid = lines[start-1].split()[0]
                if pid not in pid_set:
                    idx_set.append(0)
                    pid_set.append(pid)
                    skeleton_set.append(np.zeros((step, num_joints, 7)))
                idx2 = pid_set.index(pid)
                skeleton_set[idx2][idx_set[idx2]] = np.asarray([map(float, np.array(line_per.strip().split())[sidx]) \
                                            for line_per in lines[start+1:start+26]])
                idx_set[idx2] = idx_set[idx2] + 1
                start = start + 26
            else:
                start = start + 1
        std_set=[]
        for idx2 in xrange(len(idx_set)):
            skeleton_set[idx2] = skeleton_set[idx2][0:idx_set[idx2]]
            xm = np.abs(skeleton_set[idx2][1:idx_set[idx2],:,0] - skeleton_set[idx2][0:idx_set[idx2]-1,:,0])
            xm = xm.sum(axis=-1)
            ym = np.abs(skeleton_set[idx2][1:idx_set[idx2],:,1] - skeleton_set[idx2][0:idx_set[idx2]-1,:,1])
            ym = ym.sum(axis=-1)
            std_set.append((np.std(xm), np.std(ym)))
        return skeleton_set, pid_set, std_set

if __name__ == '__main__':
    #data_path = '/media/vhquan/APCS - Study/Thesis/Skeleton dataset/NTU RGB+D Dataset/nturgb+d_skeletons/'
    data_path = 'duongdantoifileskeleton'
    db = ntu_rgbd(data_path)
    # db.load_skeleton_file('S011C001P028R001A034.skeleton')
    db.save_h5_file_skeleton_list('data/subj_seq', [data_path], split='test')
