"""
Tensorflow implementation of the Interaction networks for the identification of boosted Higgs to bb decays https://arxiv.org/abs/1909.12285 
"""

import os
import itertools
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class InteractionModel(models.Model):
    def __init__(self, n_constituents, n_targets, params, hidden, n_vertices, params_v, vv_branch=False, De=5, Do=6):
        super(InteractionModel, self).__init__()
        self.hidden = int(hidden)
        self.P = params
        self.N = n_constituents
        self.S = params_v
        self.Nv = n_vertices
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.assign_matrices()
        self.assign_matrices_SV()
        self.vv_branch = vv_branch
        
        if self.vv_branch:
            self.assign_matrices_SVSV()
        
        self.Ra = tf.ones([self.Dr, self.Nr])
        self.fr1 = layers.Dense(self.hidden, input_shape=(2 * self.P + self.Dr,))
        self.fr2 = layers.Dense(int(self.hidden/2), input_shape=(self.hidden,))
        self.fr3 = layers.Dense(self.De, input_shape=(int(self.hidden/2),))
        self.fr1_pv = layers.Dense(self.hidden, input_shape=(self.S + self.P + self.Dr,))
        self.fr2_pv = layers.Dense(int(self.hidden/2), input_shape=(self.hidden,))
        self.fr3_pv = layers.Dense(self.De, input_shape=(int(self.hidden/2),))
        
        if self.vv_branch:
            self.fr1_vv = layers.Dense(self.hidden, input_shape=(2 * self.S + self.Dr,))
            self.fr2_vv = layers.Dense(int(self.hidden/2), input_shape=(self.hidden,))
            self.fr3_vv = layers.Dense(self.De, input_shape=(int(self.hidden/2),))
        
        self.fo1 = layers.Dense(self.hidden, input_shape=(self.P + self.Dx + (2 * self.De),))
        self.fo2 = layers.Dense(int(self.hidden/2), input_shape=(self.hidden,))
        self.fo3 = layers.Dense(self.Do, input_shape=(int(self.hidden/2),))

        if self.vv_branch:
            self.fo1_v = layers.Dense(self.hidden, input_shape=(self.S + self.Dx + (2 * self.De),))
            self.fo2_v = layers.Dense(int(self.hidden/2), input_shape=(self.hidden,))
            self.fo3_v = layers.Dense(self.Do, input_shape=(int(self.hidden/2),))

        if self.vv_branch:
            self.fc_fixed = layers.Dense(self.n_targets, input_shape=(2*self.Do,))
        else:
            self.fc_fixed = layers.Dense(self.n_targets, input_shape=(self.Do,))
        
        self.mlp = layers.Dense(n_targets)

    def assign_matrices(self):
        Rr = np.zeros([self.N, self.Nr], dtype=np.float32)
        Rs = np.zeros([self.N, self.Nr], dtype=np.float32)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            Rr[r, i]  = 1
            Rs[s, i] = 1
        self.Rr = tf.convert_to_tensor(Rr)
        self.Rs = tf.convert_to_tensor(Rs)
        del Rs, Rr

    def assign_matrices_SV(self):
        Rk = np.zeros([self.N, self.Nt], dtype=np.float32)
        Rv = np.zeros([self.Nv, self.Nt], dtype=np.float32)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv))]
        for i, (k, v) in enumerate(receiver_sender_list):
            Rk[k, i] = 1
            Rv[v, i] = 1
        self.Rk = tf.convert_to_tensor(Rk)
        self.Rv = tf.convert_to_tensor(Rv)
        del Rk, Rv

    def assign_matrices_SVSV(self):
        Rl = np.zeros([self.Nv, self.Ns], dtype=np.float32)
        Ru = np.zeros([self.Nv, self.Ns], dtype=np.float32)
        receiver_sender_list = [i for i in itertools.product(range(self.Nv), range(self.Nv)) if i[0]!=i[1]]
        for i, (l, u) in enumerate(receiver_sender_list):
            Rl[l, i] = 1
            Ru[u, i] = 1
        self.Rl = tf.convert_to_tensor(Rl)
        self.Ru = tf.convert_to_tensor(Ru) 

    def call(self, inputs, **kwargs):
        x = inputs[0]
        y = inputs[1]

        ###PF Candidate - PF Candidate###
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = tf.concat([Orr, Ors], 1)
        ### First MLP ###
        B = tf.transpose(B, perm=[0, 2, 1])
        B = tf.nn.relu(self.fr1(tf.reshape(B, [-1, 2 * self.P + self.Dr])))
        B = tf.nn.relu(self.fr2(B))
        E = tf.nn.relu(tf.reshape(self.fr3(B), [-1, self.Nr, self.De]))
        del B
        E = tf.transpose(E, perm=[0, 2, 1])
        Ebar_pp = self.tmul(E, tf.transpose(self.Rr, perm=[1, 0]))
        del E
        
        ####Secondary Vertex - PF Candidate### 
        Ork = self.tmul(x, self.Rk)
        Orv = self.tmul(y, self.Rv)
        B = tf.concat([Ork, Orv], 1)
        ### First MLP ###
        B = tf.transpose(B, perm=[0, 2, 1])
        B = tf.nn.relu(self.fr1_pv(tf.reshape(B, [-1, self.S + self.P + self.Dr])))
        B = tf.nn.relu(self.fr2_pv(B))
        E = tf.nn.relu(tf.reshape(self.fr3_pv(B), [-1, self.Nt, self.De]))
        del B
        E = tf.transpose(E, perm=[0, 2, 1])
        Ebar_pv = self.tmul(E, tf.transpose(self.Rk, perm=[1, 0]))
        Ebar_vp = self.tmul(E, tf.transpose(self.Rv, perm=[1, 0]))
        del E
 
        ###Secondary vertex - secondary vertex###
        if self.vv_branch:
            Orl = self.tmul(y, self.Rl)
            Oru = self.tmul(y, self.Ru)
            B = tf.concat([Orl, Oru], 1)
            ### First MLP ###
            B = tf.transpose(B, perm=[0, 2, 1])
            B = tf.nn.relu(self.fr1_vv(tf.reshape(B, [-1, 2 * self.S + self.Dr])))
            B = tf.nn.relu(self.fr2_vv(B))
            E = tf.nn.relu(tf.reshape(self.fr3_vv(B), [-1, self.Ns, self.De]))
            del B
            E = tf.transpose(E, perm=[0, 2, 1])
            Ebar_vv = self.tmul(E, tf.transpose(self.Rl, perm=[1, 0]))
            del E

        ####Final output matrix for particles###
        C = tf.concat([x, Ebar_pp, Ebar_pv], 1)
        del Ebar_pp
        del Ebar_pv
        C = tf.transpose(C, perm=[0, 2, 1])
        ### Second MLP ###
        C = tf.nn.relu(self.fo1(tf.reshape(C, [-1, self.P + self.Dx + (2 * self.De)])))
        C = tf.nn.relu(self.fo2(C))
        O = tf.nn.relu(tf.reshape(self.fo3(C), [-1, self.N, self.Do]))
        del C

        if self.vv_branch:
            ####Final output matrix for particles### 
            C = tf.concat([y, Ebar_vv, Ebar_vp], 1)
            del Ebar_vv
            del Ebar_vp
            C = tf.transpose(C, perm=[0, 2, 1])
            ### Second MLP ###
            C = tf.nn.relu(self.fo1_v(tf.reshape(C, [-1, self.S + self.Dx + (2 * self.De)])))
            C = tf.nn.relu(self.fo2_v(C))
            O_v = tf.nn.relu(tf.reshape(self.fo3_v(C), [-1, self.Nv, self.Do]))
            del C
        
        #Taking the sum of over each particle/vertex
        N = tf.reduce_sum(O, 1)
        del O
        if self.vv_branch:
            N_v = tf.reduce_sum(O_v, 1)
            del O_v
            return N, Nv 
        
        return self.mlp(N)

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = tf.shape(x)
        y_shape = tf.shape(y)
        return tf.reshape(tf.matmul(tf.reshape(x, [-1, x_shape[2]]), y), [-1, x_shape[1], y_shape[1]]) 
