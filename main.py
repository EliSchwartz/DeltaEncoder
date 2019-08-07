# © Copyright IBM Corp. 2019


import pickle
from deltaencoder import DeltaEncoder


########### Load Data ################
features_train, labels_train, features_test, labels_test, episodes_1shot, episodes_5shot = pickle.load(open('data/mIN.pkl','rb'))

# features_train/features_test are features extracted from some backbone (resnet18); they are np array with size = (N,D), where N is the number of samples and D the features dimensions
# labels_train/labels_test are one hot GT labels with size = (N,C), where C is the number of classes (can be different for train and test sets
# episodes_*shot are supplied for reproduction of the paper results size=(num_episodes, num_classes, num_shots, D)



######### 1-shot Experiment #########
args = {'data_set' : 'mIN',
        'num_shots' : 1,
        'num_epoch': 6,
        'nb_val_loop': 10,
        'learning_rate': 1e-5, 
        'drop_out_rate': 0.5,
        'drop_out_rate_input': 0.0,
        'batch_size': 128,
        'noise_size' : 16,
        'nb_img' : 1024,
        'num_ways' : 5,
        'encoder_size' : [8192],
        'decoder_size' : [8192],
        'opt_type': 'adam'
       }

model = DeltaEncoder(args, features_train, labels_train, features_test, labels_test, episodes_1shot)
model.train(verbose=False)



######### 5-shot Experiment #########
args = {'data_set' : 'mIN',
        'num_shots' : 5,
        'num_epoch': 12,
        'nb_val_loop': 10,
        'learning_rate': 1e-5,
        'drop_out_rate': 0.5,
        'drop_out_rate_input': 0.0,
        'batch_size': 128,
        'noise_size' : 16,
        'nb_img' : 1024,
        'num_ways' : 5,
        'encoder_size' : [8192],
        'decoder_size' : [8192],
        'opt_type': 'adam'
       }

model = DeltaEncoder(args, features_train, labels_train, features_test, labels_test, episodes_5shot)
model.train(verbose=False)
