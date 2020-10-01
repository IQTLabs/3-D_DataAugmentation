args = dict()

# Project
args['root_dir'] = '/home/mlomnitz/Documents/Projects/pose2pose'
args['log_dir'] = 'checkpoints'
# Data
args['data_dir'] = 'data/p2p'
args['meta_data'] = 'p2p_index.csv'
args['norm_mean'] = [0.5, 0.5, 0.5]
args['norm_std'] = [0.5, 0.5, 0.5]
# Training
args['n_epochs'] = 50
args['e_saves'] = 5
args['lr'] = 0.001
args['batch_size'] = 64
args['n_workers'] = 64
args['gpu'] = [0]
args['face_id'] = True
args['GAN_weight'] = 0.25
