# © Copyright IBM Corp. 2019

from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import os
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

try:
    xrange
except NameError:  # Python 3
    xrange = range


class linear_classifier(object):
    def __init__(self, features_train, labels_train, features_test, labels_test,
                 learning_rate=0.0005, number_epoch=25, batch_size=100):
        self.decay_factor = 0.9
        
        self.features_test = features_test
        self.labels_test = labels_test
        self.features_train = features_train
        self.labels_train = labels_train

        
        self.class_idx = np.where(np.sum(self.labels_train, axis=0) != 0)[0]
        self.labels_train = self.labels_train[:, self.class_idx]
        self.labels_test = self.labels_test[:, self.class_idx]
        idx = np.any(self.labels_test, axis=1)
        self.labels_test = self.labels_test[idx]
        self.features_test = self.features_test[idx]
        
        
        self.learning_rate = learning_rate
        self.number_epoch = number_epoch
        self.batch_size = batch_size
        self.features_pl = tf.placeholder(tf.float32, shape=(None, self.features_test.shape[1]))
        self.labels_pl = tf.placeholder(tf.float32, shape=(None, self.labels_test.shape[1]))
        self.batch_size_pl = tf.placeholder(tf.int32)
        self.lr_pl = tf.placeholder(tf.float32, shape=(None))
        self.model()

    def model(self):
        self.logits_op = tf.layers.dense(inputs=self.features_pl, units=self.labels_test.shape[1])
        self.softmax_op = tf.nn.softmax(self.logits_op)
        self.loss_op = self.loss(self.logits_op, self.labels_pl)
        self.train_op = self.training(self.loss_op, self.lr_pl)

    def linear(self, input, output_dim, name=None, stddev=0.02):
        with tf.variable_scope(name or 'linear'):
            norm = tf.random_normal_initializer(stddev=stddev)
            const = tf.constant_initializer(0.0)
            w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
            b = tf.get_variable('b', [output_dim], initializer=const)
            return tf.matmul(input, w) + b, b

    def loss(self, logits, labels_pl):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_pl, logits=logits, name='softmax'))
        return loss

    def training(self, loss_func, learning_rate):
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(loss_func, global_step=global_step)

    def next_batch(self, start, end):
        if start == 0:
            idx = np.r_[:self.features_train.shape[0]]
            random.shuffle(idx)
            self.features_train = self.features_train[idx]
            self.labels_train = self.labels_train[idx]
        if end > self.features_train.shape[0]:
            end = self.features_train.shape[0]
        return self.features_train[start:end], self.labels_train[start:end]

    def val(self):
        logits = self.linear_sess.run(self.softmax_op, feed_dict={self.features_pl: self.features_test,
                                                                  self.labels_pl: self.labels_test})
        acc = accuracy_score(np.argmax(logits,axis=1),np.argmax(self.labels_test,axis=1))
        return acc 
    
    def learn(self, sess):

#         self.features_test_temp = self.features_test
        self.linear_sess = sess
        init = tf.global_variables_initializer()
        self.linear_sess.run(init)
        self.learning_rate = 0.001
        best_acc = best_acc_seen = best_acc_unseen = 0.0
        last_loss_epoch = None
        for i in xrange(self.number_epoch):
            mean_loss_d = 0.0
            for count in xrange(0, self.features_train.shape[0], self.batch_size):
                features_batch, labels_batch = self.next_batch(count, count+self.batch_size)
                _, loss_value = self.linear_sess.run([self.train_op, self.loss_op],
                                                     feed_dict={self.features_pl: features_batch,
                                                                self.labels_pl: labels_batch,
                                                                self.lr_pl: self.learning_rate})
                mean_loss_d += loss_value

            mean_loss_d /= (self.features_train.shape[0] / self.batch_size)

            if last_loss_epoch is not None and mean_loss_d > last_loss_epoch:
                self.learning_rate *= self.decay_factor
            else:
                last_loss_epoch = mean_loss_d

            acc = self.val()
            if acc > best_acc:
                best_acc = acc
        return best_acc



    
class DeltaEncoder(object):
    def __init__(self, args, features, labels, features_test, labels_test, episodes, resume = ''):
        tf.reset_default_graph()
        
        self.count_data = 0
        self.num_epoch = args['num_epoch']
        self.noise_size = args['noise_size']
        self.nb_val_loop = args['nb_val_loop']
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.batch_size = args['batch_size']
        self.drop_out_rate = args['drop_out_rate']
        self.drop_out_rate_input = args['drop_out_rate_input']
        self.best_acc = 0.0
        self.name = args['data_set']
        self.last_file_name = ""
        self.nb_fake_img = args['nb_img']
        self.learning_rate = args['learning_rate']
        self.decay_factor = 0.9
        self.num_shots = args['num_shots']
        self.num_ways = args['num_ways']
        self.resume = resume
        self.save_var_dict = {}

        self.features, self.labels = features, labels
        self.features_test, self.labels_test = features_test, labels_test
        self.episodes = episodes


        self.features_dim = self.features.shape[1]    
        self.reference_features = self.random_pairs(self.features, self.labels)


        
        # discriminator input => image features
        self.x_pl = tf.placeholder(tf.float32, shape=(None, self.features_dim))
        self.z_pl = tf.placeholder(tf.float32, shape=(None, self.noise_size))
        self.reference_features_pl = tf.placeholder(tf.float32, shape=(None, self.features_dim))
        self.batch_size_pl = tf.placeholder(tf.int32)
        self.drop_out_rate_pl = tf.placeholder(tf.float32)
        self.drop_out_rate_input_pl = tf.placeholder(tf.float32)
        self.lr_pl = tf.placeholder(tf.float32, shape=(None))

        self._create_model()

    
     # assign pairs with the same labels
    def random_pairs(self,X, labels):
        Y = X.copy()
        for l in range(labels.shape[1]):
            inds = np.where(labels[:,l])[0]
            inds_pairs = np.random.permutation(inds)
            Y[inds,:] = X[inds_pairs,:]
        return Y
    
    def _create_model(self):

        with tf.variable_scope('E'):
            self.pred_noise = self.encoder(self.x_pl, self.reference_features_pl)

        with tf.variable_scope('D') as scope:
            self.pred_x = self.decoder(self.reference_features_pl, self.pred_noise)
            scope.reuse_variables()
            self.decode = self.decoder(self.reference_features_pl, self.z_pl)

        abs_diff = tf.losses.absolute_difference(self.x_pl[:,:self.features_dim],
                                                 self.pred_x,reduction=tf.losses.Reduction.NONE)
        
        k = 2.0 
        w = tf.pow(abs_diff,tf.fill([self.batch_size_pl, self.features_dim], k))
        nom = tf.reduce_sum(w,1,keepdims=True)
        nom = nom + tf.constant(1.0e-7)
        w = w / nom
        abs_diff = w * abs_diff
        
        self.loss_e = tf.reduce_mean(tf.reduce_sum(abs_diff,1))

        self.opt_e = self.optimizer(self.loss_e, self.lr_pl)

    def encoder(self, features, reference_features):
        features = tf.nn.dropout(features, 1.0-self.drop_out_rate_input_pl)
        input = tf.concat([features, reference_features], 1)
        for i, size in enumerate(self.encoder_size):
            input_lin, w, b = self.linear(input, size, name='e'+str(i))
            input = tf.nn.dropout(self.lrelu(input_lin), 1.0-self.drop_out_rate_pl)
        h, w, b = self.linear(input, self.noise_size, name='e'+str(len(self.encoder_size)))
        return h

    def decoder(self, reference_features, code):
        input = tf.concat([reference_features, code], 1)
        for i, size in enumerate(self.decoder_size):
            input_lin, w, b = self.linear(input, size, name='d'+str(i))
            input = tf.nn.dropout(self.lrelu(input_lin), 1.0-self.drop_out_rate_pl)
        h, w, b = self.linear(input, self.features.shape[1], name='d'+str(len(self.decoder_size)))
              
        return h
    
    def linear(self, input, output_dim, name=None, stddev=0.01):
        print(name)
        with tf.variable_scope(name or 'linear'):
            if self.resume:
                w_init = tf.constant(self.resume_dict[name][0])
                b_init = tf.constant(self.resume_dict[name][1])
                w = tf.get_variable('w', initializer=w_init)
                b = tf.get_variable('b', initializer=b_init)
            else:
                w_init = tf.random_normal_initializer(stddev=stddev)
                b_init = tf.constant_initializer(0.0)
                w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=w_init)
                b = tf.get_variable('b', [output_dim], initializer=b_init)
            self.save_var_dict[(name, 0)] = w
            self.save_var_dict[(name, 1)] = b
            return tf.matmul(input, w) + b, w, b

    def optimizer(self, loss, lr):
        batch = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=batch)
        return optimizer

    def next_batch(self, start, end):
        if start == 0:
            if self.num_shots:
                self.reference_features = self.random_pairs(self.features, self.labels)
            idx = np.r_[:self.features.shape[0]]
            random.shuffle(idx)
            self.features = self.features[idx]
            self.reference_features = self.reference_features[idx]
            self.labels = self.labels[idx]
        if end > self.features.shape[0]:
            end = self.features.shape[0]
        return self.features[start:end], self.reference_features[start:end], self.labels[start:end]

    def train(self, verbose=False):
        with tf.Session() as self.session:
            tf.global_variables_initializer().run()
            last_loss_epoch = None
            acc = self.val()
            print('Unseen classes accuracy without training: {}'.format(acc)) 
            print("-----")
            for epoch in xrange(self.num_epoch):
                mean_loss_e = 0.0
                for count in xrange(0, self.features.shape[0], self.batch_size):
                    features_batch, reference_features_batch, labels_batch = self.next_batch(count, count+self.batch_size)
            
                    # update discriminator
                    loss_e, _ = self.session.run([self.loss_e, self.opt_e], {
                                    self.x_pl: features_batch,
                                    self.reference_features_pl: reference_features_batch,
                                    self.batch_size_pl: features_batch.shape[0],
                                    self.drop_out_rate_input_pl: self.drop_out_rate_input,
                                    self.drop_out_rate_pl: self.drop_out_rate,
                                    self.lr_pl: self.learning_rate})
                    mean_loss_e += loss_e
                
                    c = (count/self.batch_size)+1
                    if verbose:
                        if np.mod(c,10)==1:
                            print('Batch#{0} Loss {1}'.format(c,mean_loss_e/(c+1e-7)))


                mean_loss_e /= (self.features.shape[0] / self.batch_size)
                if verbose:
                    print('epoch : {}: E : {}'.format(epoch, mean_loss_e))
                if last_loss_epoch is not None and mean_loss_e > last_loss_epoch:
                    self.learning_rate *= self.decay_factor
                    if verbose:
                        print("AE learning rate decay: ", self.learning_rate)
                else:
                    last_loss_epoch = mean_loss_e
                    
                acc = self.val()
                if acc > self.best_acc:
                    if self.best_acc != 0.0:
                        os.remove(self.last_file_name + ".npy")
                    self.best_acc = acc
                    self.last_file_name = "model_weights/" + self.name  + '_' \
                                            + str(self.num_shots) + '_shot_' \
                                            + str(np.around(self.best_acc, decimals=2)) + '_acc'      
                    self.save_npy(self.session, self.last_file_name)
                    print('epoch {}: Higher unseen classes accuracy reached: {} (Saved in {}.npy)'.format(epoch+1, acc, self.last_file_name))
                else:
                    print('epoch {}: Lower unseen classes accuracy reached: {} (<={})'.format(epoch+1, acc,self.best_acc))    
                print("-----")
            self.session.close()
            return self.best_acc
        
    def generate_samples(self, reference_features_class, labels_class, nb_ex):
        iterations = 0
        features = np.zeros((nb_ex * labels_class.shape[0], self.features.shape[1]))
        labels = np.zeros((nb_ex * labels_class.shape[0], labels_class.shape[1]))
        reference_features = np.zeros((nb_ex * labels_class.shape[0], self.reference_features.shape[1]))
        for c in xrange(labels_class.shape[0]):
            if True: #sample "noise" from training set
                inds = np.random.permutation(xrange(self.features.shape[0]))[:nb_ex]
                noise = self.session.run(self.pred_noise, {
                            self.x_pl:  self.features[inds,...],
                            self.reference_features_pl:  self.reference_features[inds,...],
                            self.drop_out_rate_input_pl: 0.0,
                            self.drop_out_rate_pl: 0.0})
            else:
                noise = np.random.normal(0, 1, (nb_ex, self.noise_size))
                                
            features[c * nb_ex:(c * nb_ex) + nb_ex] = self.session.run(self.decode, {
                self.z_pl:  noise,
                self.reference_features_pl: np.tile(reference_features_class[c], (nb_ex, 1)),
                self.drop_out_rate_input_pl: 0.0,
                self.drop_out_rate_pl: 0.0})
            labels[c * nb_ex:(c * nb_ex) + nb_ex] = np.tile(labels_class[c], (nb_ex, 1))
            reference_features[c * nb_ex:(c * nb_ex) + nb_ex] = np.tile(reference_features_class[c], (nb_ex, 1))
        return features, reference_features, labels
    

    def val(self, verbose = False):
        acc = []
               
        for episode_data in self.episodes:
            unique_labels_episode = episode_data[1][:,0,:]
            
            
            features, reference_features, labels = [], [], []
            for shot in range(max(self.num_shots,1)):
                unique_reference_features_test = episode_data[0][:,shot,:]
                features_, reference_features_, labels_ = self.generate_samples(unique_reference_features_test,
                                                                            unique_labels_episode,
                                                                            self.nb_fake_img/max(self.num_shots,1))
                features.append(unique_reference_features_test)
                reference_features.append(unique_reference_features_test)
                labels.append(unique_labels_episode)
                features.append(features_)
                reference_features.append(reference_features_)
                labels.append(labels_)
                if verbose:
                    print(np.mean([np.linalg.norm(x) for x in unique_reference_features_test]))
                    print(np.mean([np.linalg.norm(x) for x in features_]))
                
            features = np.concatenate(features)  
            reference_features = np.concatenate(reference_features)   
            labels = np.concatenate(labels)
            lin_model = linear_classifier(features, labels, self.features_test,
                                                     self.labels_test)
            with tf.Session() as linear_sess:
                acc_ = lin_model.learn(linear_sess)
                acc.append(acc_)

       
        acc = 100*np.mean(acc)                
        return acc

    def lrelu(self, x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def save_npy(self, sess, npy_path):
        assert isinstance(sess, tf.Session)
        data_dict = {}
        for (name, idx), var in self.save_var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        return npy_path




