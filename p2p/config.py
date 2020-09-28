args = dict()

# Project
args['root_dir'] = '/home/mlomnitz/Pose2Pose'
args['log_dir'] = 'checkpoints'
# Data
args['data_dir'] = 'data/p2p'
args['meta_data'] = 'p2p_index.csv'
args['norm_mean'] = [0.5, 0.5, 0.5]
args['norm_std'] = [0.5, 0.5, 0.5]
# Training
args['n_epochs'] = 50
args['e_saves'] = 5
args['lr'] = 0.0005
args['batch_size'] = 96
args['n_workers'] = 32
args['gpu'] = [1, 2, 3]
args['face_id'] = True
