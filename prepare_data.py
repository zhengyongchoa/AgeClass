from __future__ import print_function
import numpy as np
from scipy import signal
import pickle
import _pickle as cPickle
import time , csv ,  h5py , os
import soundfile , librosa
import options as opt
import threading

# Read wav
def read_audio(path, target_fs=None):
    # (audio, fs)  = librosa.load( path )
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

# Write wav
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

# Create an empty folder
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)



def multithread(wav_dir , out_dir ):

    with open(wav_dir) as f:
        data_list = [line for line in f]
    L = len(data_list)
    ind = np.linspace(0 , L , num =6)
    ind =  [int(ele) for ele in ind]
    wavlist =[]
    for i in range(5):
        listname = data_list[ind[i]: ind[i+1]]
        wavlist.append(listname)

    print(0)
    # extract_features_plus(wavlist[0], out_dir)
    thread1 = threading.Thread(name='t1', target=extract_features_plus, args=(wavlist[0], out_dir))
    thread2 = threading.Thread(name='t2', target=extract_features_plus, args=(wavlist[1],  out_dir))
    thread3 = threading.Thread(name='t2', target=extract_features_plus, args=(wavlist[2],  out_dir))
    thread4 = threading.Thread(name='t2', target=extract_features_plus, args=(wavlist[3],  out_dir))
    thread5 = threading.Thread(name='t2', target=extract_features_plus, args=(wavlist[4],  out_dir))

    thread1.start()  # 启动线程1
    thread2.start()  # 启动线程2
    thread3.start()  # 启动线程2
    thread4.start()  # 启动线程2
    thread5.start()  # 启动线程2


def extract_features_plus(wavlist , out_dir):
    fs = opt.sample_rate

    create_folder(out_dir)
    names = sorted(wavlist)
    print("音频数据条数: %d" % len(names))

    cnt = 0
    t1 = time.time()
    for na in names:
        wav_path = na.split()[0]
        label = na.split()[1]
        out_path = out_dir + '/' + label + '_' + wav_path.split('/')[-1].split('.')[0] + '.p'

        # Skip features already computed
        recompute =1
        if recompute or (not os.path.isfile(out_path)):
            # print(cnt, out_path)

            (wavdata, _) = read_audio(wav_path, fs)
            # Skip corrupted wavs
            if wavdata.shape[0] < 3 * fs:
                print("File %s is too short!" % wav_path)
            else:
                audio = wavdata[fs: 3 * fs]
                outfea = singlefeature(audio)
                cPickle.dump(outfea, open(out_path, 'wb'), -1)
        cnt += 1
    print("抽取特征耗时！: %s" % (time.time() - t1,))

def extract_features(wav_dir, out_dir, recompute):

    fs = opt.sample_rate
    n_window = opt.n_window
    n_overlap = opt.n_overlap

    create_folder(out_dir)
    names = [na for na in os.listdir(os.path.join( opt.workspace, wav_dir ) ) if na.endswith(".wav")]
    names = sorted(names)
    print("音频数据条数: %d" % len(names))
    label = wav_dir.split('/')[-1]


    cnt = 0
    t1 = time.time()
    for na in names:
        wav_path = opt.workspace +'/'+  wav_dir + '/' + na
        out_path = out_dir + '/' + label + '_' + os.path.splitext(na)[0] + '.p'

        # Skip features already computed
        if recompute or (not os.path.isfile(out_path)):
            print(cnt, out_path)

            (wavdata, _) = read_audio(wav_path, fs)
            # Skip corrupted wavs
            if wavdata.shape[0] < 3*fs:
                print("File %s is too short!" % wav_path)
            else:
                audio = wavdata[fs : 3*fs]
                outfea = singlefeature(audio)
                cPickle.dump(outfea, open(out_path, 'wb'), -1)
        cnt += 1
    print("抽取特征耗时！: %s" % (time.time() - t1,))

def singlefeature(audio):
    fs = opt.sample_rate
    n_window = opt.n_window
    # 特征提取
    # 特征1： spectrogram

    # 提取特征2
    # f2 ,voiced_flag, voiced_probs= librosa.pyin( audio, sr = fs, frame_length = n_window ,hop_length =512, center=False, fmin = 65 ,fmax = 2000 )
    f2, voiced_flag, voiced_probs = librosa.pyin(audio, sr=fs, frame_length=n_window, hop_length=512, fmin=65,
                                                 fmax=2000)
    # voiced_flag =np.array(  [int(elem) for elem in voiced_flag] )

    # 特区特征3
    f3 = librosa.feature.mfcc(audio, win_length=1024, sr=fs, hop_length=512)

    # 特征优化：
    for i in range(len(voiced_flag)):
        if not voiced_flag[i]:
            f2[i] = 0
            f3[:, i] = 0

    # 特征拼接20*63 +1*63
    f2 = f2[np.newaxis, :]
    outfea = np.concatenate((f2, f3))

    return outfea


def make_csv( dir , modetype ) :
    fe_dir = os.path.join(dir , modetype)
    f_all = []
    names = os.listdir(fe_dir)
    names = sorted(names)
    for fe_na in names:
        fe_path = os.path.join(fe_dir, fe_na)
        fe_path = [fe_path]
        f_all.append(fe_path)

    out = modetype + '.csv'
    out_path = os.path.join( dir , out )
    with open( out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in f_all :
            writer.writerow(row)

    print("Save csv to %s" % out_path)


def load_csv(csv_path):
    filpath =[]
    if csv_path != "":  # Pack from csv file (training & testing from dev. data)
        # with open(csv_path, 'rb') as f:
        #     reader = csv.reader(f)
        #     lis = list(reader)
        f = open(csv_path)
        data = csv.reader(f)  # ①
        for line in data:
            tmp = line
            filpath.append(tmp)

    else:
        print('没有特征文件！')

    return filpath

### Main function
if __name__ == '__main__':

    mode = ['train', 'valid', 'test']  # 选择 训练 ，测试， 验证文件夹
    for i in mode:
        make_csv(opt.feat_dir, i)


    # category = 'train'
    # feat_dir = os.path.join(opt.feat_dir, category)
    # create_folder(feat_dir)
    # wavlist = category + '_list.txt'
    # wav_dir = os.path.join(opt.audio_dir, wavlist)
    #
    # if 1: #
    #     with open(wav_dir) as f:
    #         data_list = [line for line in f]
    #     extract_features_plus(data_list, feat_dir)
    #
    # else:  # release模式，多线进行特征提取,线程n > core。
    #     multithread(wav_dir ,  feat_dir)



    make_csv(opt.feat_dir, category)
