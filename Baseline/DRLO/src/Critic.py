import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, add, Dropout
import numpy as np

HIDDEN1_UNITS = 1000
HIDDEN2_UNITS = 800

class Critic:
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)

    def create_model(self):
        state_input = Input((self.state_dim,))
        s1 = Dense(HIDDEN1_UNITS, activation='relu')(state_input)
        s2 = Dense(HIDDEN2_UNITS, activation='linear')(s1)
        action_input = Input((self.action_dim,))
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(action_input)
        c1 = add([s2, a1])
        c2 = Dense(HIDDEN2_UNITS, activation='relu')(c1)
        output = Dense(1, activation='linear')(c2)
        return tf.keras.Model([state_input, action_input], output)

    def q_grads(self, states, actions):
        actions = tf.convert_to_tensor(actions, dtype=tf.float64)
        states = tf.convert_to_tensor(states, dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model([states, actions])
            q_values = tf.squeeze(q_values, axis=-1)
        return tape.gradient(q_values, actions)

    def compute_loss(self, q_pred, q_target):
        mse = tf.losses.mse(y_true=q_target, y_pred=q_pred)
        loss = tf.reduce_mean(mse)
        return loss

    def train(self, states, actions, q_target, learning_rate):
        states = tf.convert_to_tensor(states, dtype=tf.float64)
        actions = tf.convert_to_tensor(actions, dtype=tf.float64)

        with tf.GradientTape() as tape:
            q_pred = self.model([states, actions], training=True)
            loss = self.compute_loss(q_pred, tf.stop_gradient(q_target))
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss