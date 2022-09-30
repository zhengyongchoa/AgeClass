device_mode = 'CPU'  #  训练时 GPU or CPU
test_device = 'CPU'  # 模型测试过程中使用 GPU or CPU
max_epoch = 15
lr = 1e-2
num_workers = 4
display = 10
nClasses = 2
savemodel_dir = 'models'
batch_size= 34  # 64
test_batch_size = 1

# 音频所在的路径
workspace = "/home/zyc/zycdata/ageRecognition"
wav_dir = 'Agedata3'
audio_dir = 'wav_2classify'
# 特征所在路径
feat_dir ='feature_2classify_3'

train_file = './feature_2classify_3/train.csv'
valid_file = './feature_2classify_3/valid.csv'
test_file =  './feature_2classify_3/test.csv'

#特征的参数
sample_rate = 16000
n_window = 1024
n_overlap = 512      # ensure 240 frames in 10 seconds
max_len = 240        # sequence max length is 10 s, 240 frames.
step_time_in_sec = float(n_window - n_overlap) / sample_rate

n_time = 21
n_freq = 63