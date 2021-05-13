#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, argparse, tqdm
import time
start = time.time()
from argparse import RawTextHelpFormatter

start = time.time()

base_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='SPOT-RNA-1D: RNA backbone torsion and pseudotorsion angle prediction using dilated convolutional neural networks.', usage='use "./%(prog)s --help" for more information', formatter_class=RawTextHelpFormatter)
parser.add_argument('--seq_file',default=base_path + '/datasets/TS1_seqs.fasta', type=str, help="Path to the input sequence fasta file\n" "default file: " + base_path + "/datasets/TS1_seqs.fasta", metavar='')
parser.add_argument('--save_outputs',default=base_path + '/outputs', type=str, help="Path to the folder for saving output\n" "default folder: " + base_path + "/outputs", metavar='')
parser.add_argument('--batch_size',default=10, type=int, help="Number of simultaneous prediction for multi sequence fasta file input\n" "default batch size: 10", metavar='')
parser.add_argument('--gpu', default=-1, type=int, help="To run on GPU, specify GPU number. If only one GPU in the system specify 0\n" "default: -1 (no GPU)\n", metavar='')
parser.add_argument('--cpu',default=16, type=int, help="Specify number of cpu threads that SPOT-RNA-1D can use\n" "default = 16\n\n", metavar='')
args = parser.parse_args()

class bcolors:
    RED   = "\033[1;31m"
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"


########## mean and standard deviation of trainin data TR1 ####################
norm_mu = [0.24416392, 0.19836862, 0.30642843, 0.24948534]
norm_std = [0.43031312, 0.39954973, 0.46168336, 0.43343267]


print(bcolors.CYAN + '\nReading input seqeunce file from ' + args.seq_file + bcolors.RESET)

############## read input file #####################
with open(args.seq_file) as file:
    input_data = [line.strip() for line in file.read().splitlines() if line.strip()]
count = int(len(input_data)/2)  # count number of input sequences

ids = [''.join(e if e.isalnum() else '_' for e in input_data[2*i].replace(">", "")) for i in range(count)]   # name of each input sequence after ignoring by special characters.

###### extracting all the input sequences  #############
sequences = {}
for i,I in enumerate(ids):
    sequences[I] = input_data[2*i+1].replace(" ", "").upper().replace("T", "U")  # removing spaces, converting lowercase to uppercase, and replacing T with U

bases = np.array([base for base in 'AUGC'])

##### one-hot encoding of the input sequence and saving into a dictionary (feat_dic) #########
feat_dic = {}
for i,I in enumerate(ids):
	feat_dic[I] = np.concatenate([[(bases==base.upper()).astype(int)] if str(base).upper() in 'AUGC' else np.array([[0]*4]) for base in sequences[I]])  # one-hot encoding Lx4


os.environ['KMP_WARNINGS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if args.gpu == -1:
	config = tf.ConfigProto(intra_op_parallelism_threads=args.cpu, inter_op_parallelism_threads=args.cpu)
else:
	config = tf.compat.v1.ConfigProto()
	config.allow_soft_placement=True
	config.log_device_placement=False


def get_data(sample_feat, ids, batch_size, i, norm_mu,norm_std):
	###############################################
	# prepare normalize input feature
	# make batch of input features
    # prepare zero-mask for different length input sequences
	###############################################
    data = [(sample_feat[j][:,:]-norm_mu)/norm_std for j in ids[i * batch_size:np.min([(i + 1) * batch_size, len(ids)])]] 
    data = [np.concatenate([np.ones((j.shape[0], 1)), j], axis=1) for j in data]
    seq_lens = [j.shape[0] for j in data]
    batch_ids = [j for j in ids[i * batch_size:np.min([(i + 1) * batch_size, len(ids)])]]
    max_seq_len = max(seq_lens)
    data = np.concatenate([np.concatenate([j, np.zeros((max_seq_len - j.shape[0], j.shape[1]))])[None, :, :] for j in data])

    mask = np.concatenate([np.concatenate([np.ones((1, seq_lens[j])), np.zeros((1, max_seq_len - seq_lens[j]))], axis=1) for j in range(len(ids[i * batch_size:np.min(((i + 1) * batch_size, len(ids)))]))])

    return data, mask, seq_lens, batch_ids

def sigmoid(x):
  return x#1 / (1 + np.exp(-x))


print(bcolors.GREEN + '\nRunning SPOT-RNA-1D\n' + bcolors.RESET)

outputs = {}
with tf.compat.v1.Session(config=config) as sess:
    saver = tf.compat.v1.train.import_meta_graph(base_path + '/checkpoints/tensorflow_model.meta')
    saver.restore(sess,base_path + '/checkpoints/tensorflow_model')
    graph = tf.compat.v1.get_default_graph()
    tmp_out = graph.get_tensor_by_name('output_FC/fully_connected/BiasAdd:0')

    for batch_test in tqdm.tqdm(range(max(1,int(np.ceil(count/args.batch_size))))):
        feature, mask, seq_lens, batch_ids = get_data(feat_dic, ids, batch_size=args.batch_size, i=batch_test, norm_mu=norm_mu, norm_std=norm_std)
        out = sess.run([tmp_out],feed_dict={'input_feature:0': feature, 'seq_lens:0':seq_lens, 'zero_mask:0':mask, 'keep_prob:0':1.})
        pred_angles = sigmoid(out[0])
        for i, id in enumerate(batch_ids):
            outputs[id] = pred_angles[i,0:seq_lens[i]]
tf.compat.v1.reset_default_graph()

for id in ids:

	seq = np.array([[i,I] for i,I in enumerate(sequences[id])])

	preds = outputs[id] * 2 - 1
	preds_angle_rad = [np.arctan2(preds[:,2*i], preds[:,2*i+1]) for i in range(int(preds.shape[1]/2))]
	preds_angle = np.round(np.transpose(np.degrees(preds_angle_rad)), 2)
	final_output =  np.concatenate((seq, preds_angle), axis=1)
	np.savetxt(os.path.join(args.save_outputs, str(id.split('.')[0]))+'.txt', (final_output), delimiter='\t\t', fmt="%s", header='No.\t   Seq\t\t Alpha\t\t Beta\t\tGamma\t\tDelta\t\t Epsilon\t Zeta\t\t  Chi\t\t  Eta\t\t Theta' + '\n', comments='')

print(bcolors.GREEN + '\nFinished!' + bcolors.RESET)
print(bcolors.GREEN + 'SPOT-RNA-1D prediction saved in folder ' + args.save_outputs + bcolors.RESET)

end = time.time()
print(bcolors.CYAN + '\nProcesssing Time {:.4f} seconds\n'.format(end - start) + bcolors.RESET)

