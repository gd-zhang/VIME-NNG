from __future__ import print_function
import numpy as np
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc import ext
from collections import OrderedDict
import theano
import theano.tensor.slinalg as slinalg
import theano.tensor.nlinalg as nlinalg
import pdb

# ----------------
BNNNG_LAYER_TAG = 'BNNNGLayer'
USE_REPARAMETRIZATION_TRICK = False
# ----------------


class BNNNGLayer(lasagne.layers.Layer):
    """Probabilistic layer that uses Gaussian weights.

    Each weight has two parameters: mean and standard deviation (std).
    """

    def __init__(self,
                 incoming,
                 num_units,
                 num_data,
                 n_samples,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 prior_sd=None,
                 **kwargs):
        super(BNNNGLayer, self).__init__(incoming, **kwargs)

        self._srng = RandomStreams()

        # Set vars.
        self.nonlinearity = nonlinearity
        # print('input shape', self.input_shape)
        self.num_inputs = self.input_shape[-1] + 1
        self.num_units = num_units
        self.prior_sd = prior_sd
        self.num_data = num_data
        self.n_samples = n_samples

        # Here we set the priors.
        # -----------------------
        self.mu = self.add_param(
            lasagne.init.GlorotNormal(),
            (self.num_inputs, self.num_units),
            name='mu'
        )
        # TODO: change init
        tmp = 0.1 * np.random.normal(size=[self.num_inputs, self.num_inputs])
        self.fisher_u = self.add_param(
            np.eye(self.num_inputs) + np.matmul(tmp, tmp.transpose()),
            (self.num_inputs, self.num_inputs),
            name='fisher_u'
        )
        tmp = 0.1 * np.random.normal(size=[self.num_units, self.num_units])
        self.fisher_v = self.add_param(
            np.eye(self.num_units) + np.matmul(tmp, tmp.transpose()),
            (self.num_units, self.num_units),
            name='fisher_v'
        )
        # -----------------------

        # Backup params for KL calculations.
        self.mu_old = self.add_param(
            np.zeros((self.num_inputs, self.num_units)),
            (self.num_inputs, self.num_units),
            name='mu_old',
            trainable=False,
            oldparam=True
        )
        tmp = 0.1 * np.random.normal(size=[self.num_inputs, self.num_inputs])
        self.fisher_u_old = self.add_param(
            np.eye(self.num_inputs) + np.matmul(tmp, tmp.transpose()),
            (self.num_inputs, self.num_inputs),
            name='u_old',
            trainable=False,
            oldparam=True
        )
        tmp = 0.1 * np.random.normal(size=[self.num_units, self.num_units])
        self.fisher_v_old = self.add_param(
            np.eye(self.num_units) + np.matmul(tmp, tmp.transpose()),
            (self.num_units, self.num_units),
            name='v_old',
            trainable=False,
            oldparam=True
        )

    def get_u(self):
        pi = T.sqrt(T.mean(T.diag(self.fisher_u)) / T.mean(T.diag(self.fisher_v)))
        coeff = 1. / (self.num_data * self.prior_sd ** 2)
        u = nlinalg.matrix_inverse(self.fisher_u + pi * (coeff ** 0.5) * T.eye(self.num_inputs))
        return u / (self.num_data ** 0.5)

    def get_v(self):
        pi = T.sqrt(T.mean(T.diag(self.fisher_u)) / T.mean(T.diag(self.fisher_v)))
        coeff = 1. / (self.num_data * self.prior_sd ** 2)
        v = nlinalg.matrix_inverse(self.fisher_v + 1. / pi * (coeff ** 0.5) * T.eye(self.num_units))
        return v / (self.num_data ** 0.5)

    def get_u_old(self):
        pi = T.sqrt(T.mean(T.diag(self.fisher_u_old)) / T.mean(T.diag(self.fisher_v_old)))
        coeff = 1. / (self.num_data * self.prior_sd ** 2)
        u = nlinalg.matrix_inverse(self.fisher_u_old + pi * (coeff ** 0.5) * T.eye(self.num_inputs))
        return u / (self.num_data ** 0.5)

    def get_v_old(self):
        pi = T.sqrt(T.mean(T.diag(self.fisher_u_old)) / T.mean(T.diag(self.fisher_v_old)))
        coeff = 1. / (self.num_data * self.prior_sd ** 2)
        v = nlinalg.matrix_inverse(self.fisher_v_old + 1. / pi * (coeff ** 0.5) * T.eye(self.num_units))
        return v / (self.num_data ** 0.5)

    # TODO: support multiple samples
    def get_W(self):
        # Here we generate random epsilon values from a normal distribution
        epsilon = self._srng.normal(size=(self.n_samples, self.num_inputs, self.num_units), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable
        # Here we calculate the cholesky decomposition of u and v
        u = self.get_u()
        v = self.get_v()
        uc = slinalg.cholesky(u)
        vc = slinalg.cholesky(v)
        uc = T.tile(T.shape_padleft(uc), [self.n_samples, 1, 1])
        vc = T.tile(T.shape_padleft(T.transpose(vc)), [self.n_samples, 1, 1])
        W = self.mu + T.batched_dot(T.batched_dot(uc, epsilon), vc)
        self.W = W
        return W

    def get_output_for_reparametrization(self, input, **kwargs):
        """Implementation of the local reparametrization trick.

        This essentially leads to a speedup compared to the naive implementation case.
        Furthermore, it leads to gradients with less variance.

        References
        ----------
        Kingma et al., "Variational Dropout and the Local Reparametrization Trick", 2015
        """
        pass

    def save_old_params(self):
        """Save old parameter values for KL calculation."""
        self.mu_old.set_value(self.mu.get_value())
        self.fisher_u_old.set_value(self.fisher_u.get_value())
        self.fisher_v_old.set_value(self.fisher_v.get_value())

    def reset_to_old_params(self):
        """Reset to old parameter values for KL calculation."""
        self.mu.set_value(self.mu_old.get_value())
        self.fisher_u.set_value(self.fisher_u_old.get_value())
        self.fisher_v.set_value(self.fisher_v_old.get_value())

    def kl_div_mvg_mf(self, q_mean, q_u, q_v, p_mean, p_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a matrix variate Gaussian"""
        uc = slinalg.cholesky(q_u)
        vc = slinalg.cholesky(q_v)
        logdet_u = T.sum(2. * T.log(T.diag(uc)))
        logdet_v = T.sum(2. * T.log(T.diag(vc)))
        kl = T.sum(T.log(T.square(p_std))) - self.num_inputs * logdet_v - self.num_units * logdet_u

        diag_u = T.diag(q_u)
        diag_v = T.diag(q_v)
        diag_uv = T.outer(diag_u, diag_v)
        kl = kl - self.num_inputs * self.num_units + T.sum(diag_uv / T.square(p_std))
        kl = kl + T.sum(T.square(p_mean - q_mean) / T.square(p_std))
        return kl * 0.5

    def kl_div_mvg_mvg(self, q_mean, q_u, q_v, p_mean, p_u, p_v):
        q_uc = slinalg.cholesky(q_u)
        q_vc = slinalg.cholesky(q_v)
        logdet_qu = T.sum(2. * T.log(T.diag(q_uc)))
        logdet_qv = T.sum(2. * T.log(T.diag(q_vc)))

        p_uc = slinalg.cholesky(p_u)
        p_vc = slinalg.cholesky(p_v)
        logdet_pu = T.sum(2. * T.log(T.diag(p_uc)))
        logdet_pv = T.sum(2. * T.log(T.diag(p_vc)))

        kl = self.num_inputs * logdet_pv + self.num_units * logdet_pu - \
             self.num_inputs * logdet_qv - self.num_units * logdet_qu
        kl = kl - self.num_inputs * self.num_units
        kl = kl + T.dot(nlinalg.matrix_inverse(p_u), q_u).trace() * T.dot(nlinalg.matrix_inverse(p_v), q_v).trace()
        mean_diff = q_mean - p_mean
        kl = kl + T.sum(mean_diff * T.dot(p_u, T.dot(mean_diff, p_v)))

        return kl * 0.5

    def kl_div_new_old(self):
        kl_div = self.kl_div_mvg_mvg(
            self.mu, self.get_u(), self.get_v(), self.mu_old, self.get_u_old(), self.get_v_old())

        return kl_div

    def kl_div_old_new(self):
        return NotImplementedError

    def kl_div_new_prior(self):
        kl_div = self.kl_div_mvg_mf(
            self.mu, self.get_u(), self.get_v(), 0., self.prior_sd)

        return kl_div

    def kl_div_old_prior(self):
        kl_div = self.kl_div_mvg_mf(
            self.mu_old, self.get_u_old(), self.get_v_old(), 0., self.prior_sd)

        return kl_div

    def get_output_for(self, input, **kwargs):
        if USE_REPARAMETRIZATION_TRICK:
            return self.get_output_for_reparametrization(input, **kwargs)
        else:
            return self.get_output_for_default(input, **kwargs)

    def get_output_for_default(self, input, **kwargs):
        if input.ndim == 2:
            input = T.tile(T.shape_padleft(input), (self.n_samples, 1, 1))
        input = T.concatenate([input, T.ones_like(input[:, :, :1])], axis=2)
        self.activation = input
        pre_activation = T.batched_dot(input, self.get_W())
        self.pre_activation = pre_activation

        return self.nonlinearity(pre_activation)

    def get_output_shape_for(self, input_shape):
        return (self.n_samples, input_shape[0], self.num_units)


class BNN(LasagnePowered, Serializable):
    """Bayesian neural network (BNN) based on Blundell2016."""

    def __init__(self, n_in,
                 n_hidden,
                 n_out,
                 layers_type,
                 n_batches=50,
                 trans_func=lasagne.nonlinearities.rectify,
                 out_func=lasagne.nonlinearities.linear,
                 n_samples=10,
                 prior_sd=0.5,
                 batch_size=100,
                 use_reverse_kl_reg=False,
                 reverse_kl_reg_factor=0.1,
                 likelihood_sd=5.0,
                 second_order_update=False,
                 learning_rate=0.0001,
                 true_fisher=True,
                 compression=False,
                 information_gain=True,
                 ):

        Serializable.quick_init(self, locals())
        assert len(layers_type) == len(n_hidden) + 1

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.transf = trans_func
        self.outf = out_func
        self.n_samples = n_samples
        self.prior_sd = prior_sd
        self.layers_type = layers_type
        self.n_batches = n_batches
        self.num_data = n_batches
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        self.likelihood_sd = likelihood_sd
        self.second_order_update = second_order_update
        self.learning_rate = learning_rate
        self.true_fisher = true_fisher
        self.compression = compression
        self.information_gain = information_gain
        self.alpha = learning_rate
        self.beta = learning_rate * 0.1

        assert self.information_gain or self.compression

        # Build network architecture.
        self.build_network()

        # Build model might depend on this.
        LasagnePowered.__init__(self, [self.network])

        # Compile theano functions.
        self.build_model()

    def save_old_params(self):
        layers = list(filter(lambda l: l.name == BNNNG_LAYER_TAG,
                        lasagne.layers.get_all_layers(self.network)[1:]))
        for layer in layers:
            layer.save_old_params()

    def reset_to_old_params(self):
        layers = list(filter(lambda l: l.name == BNNNG_LAYER_TAG,
                        lasagne.layers.get_all_layers(self.network)[1:]))
        for layer in layers:
            layer.reset_to_old_params()

    def compression_improvement(self):
        """KL divergence KL[old_param||new_param]"""
        layers = list(filter(lambda l: l.name == BNNNG_LAYER_TAG,
                        lasagne.layers.get_all_layers(self.network)[1:]))
        return sum(l.kl_div_old_new() for l in layers)

    def inf_gain(self):
        """KL divergence KL[new_param||old_param]"""
        layers = list(filter(lambda l: l.name == BNNNG_LAYER_TAG,
                        lasagne.layers.get_all_layers(self.network)[1:]))
        return sum(l.kl_div_new_old() for l in layers)

    def surprise(self):
        surpr = 0.
        if self.compression:
            surpr += self.compression_improvement()
        if self.information_gain:
            surpr += self.inf_gain()
        return surpr

    def kl_div(self):
        """KL divergence KL[new_param||old_param]"""
        layers = list(filter(lambda l: l.name == BNNNG_LAYER_TAG,
                        lasagne.layers.get_all_layers(self.network)[1:]))
        return sum(l.kl_div_new_old() for l in layers)

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        layers = list(filter(lambda l: l.name == BNNNG_LAYER_TAG,
                        lasagne.layers.get_all_layers(self.network)[1:]))
        return sum(l.kl_div_new_prior() for l in layers)

    def reverse_log_p_w_q_w_kl(self):
        """KL divergence KL[p(w)||q_\phi(w)]"""
        layers = list(filter(lambda l: l.name == BNNNG_LAYER_TAG,
                        lasagne.layers.get_all_layers(self.network)[1:]))
        return sum(l.kl_div_prior_new() for l in layers)

    def get_internel_results(self):
        layers = list(filter(lambda l: l.name == BNNNG_LAYER_TAG,
                        lasagne.layers.get_all_layers(self.network)[1:]))
        activations = [l.activation for l in layers]
        ss = [l.pre_activation for l in layers]
        ws = [l.W for l in layers]

        return activations, ss, ws

    def _log_prob_normal(self, input, mu=0., sigma=1.):
        log_normal = - \
            T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - \
            T.square(input - mu) / (2 * T.square(sigma))
        return T.sum(log_normal, axis=2)

    def pred_sym(self, input):
        return lasagne.layers.get_output(self.network, input)

    def log_py_xw(self, input, target):
        # Make prediction.
        prediction = self.pred_sym(input)
        target = T.tile(T.shape_padleft(target), (self.n_samples, 1, 1))
        log_p_D_given_w = self._log_prob_normal(target, prediction, self.likelihood_sd)
        return log_p_D_given_w, prediction

    def get_updates(self, layers, ws, w_grads, activations, s_grads):
        # TODO: use alpha and beta as learning rate
        updates = OrderedDict()

        for l, w, w_grad, a, s_grad in zip(layers, ws, w_grads, activations, s_grads):
            coeff = 1./(self.num_data * l.prior_sd ** 2)
            u = l.get_u() * (self.num_data ** 0.5)
            v = l.get_v() * (self.num_data ** 0.5)
            grad_mu = T.dot(u, T.dot(T.mean(w_grad - coeff * w, axis=0), v))
            updates[l.mu] = l.mu + self.alpha * grad_mu

            a_t = a.dimshuffle((0, 2, 1))
            grad_u = T.mean(T.batched_dot(a_t, a), axis=0) / T.shape(a)[1]
            updates[l.fisher_u] = (1 - self.beta) * l.fisher_u + self.beta * grad_u

            s_grad_t = s_grad.dimshuffle((0, 2, 1))
            grad_v = T.mean(T.batched_dot(s_grad_t, s_grad), axis=0) / T.shape(s_grad)[1]
            updates[l.fisher_v] = (1 - self.beta) * l.fisher_v + self.beta * grad_v

        return updates

    def build_network(self):

        # Input layer
        network = lasagne.layers.InputLayer(shape=(1, self.n_in))

        # Hidden layers
        for i in range(len(self.n_hidden)):
            # Probabilistic layer (1) or deterministic layer (0).
            if self.layers_type[i] == 1:
                network = BNNNGLayer(
                    network, self.n_hidden[i], num_data=self.n_batches,
                    n_samples=self.n_samples, nonlinearity=self.transf,
                    prior_sd=self.prior_sd, name=BNNNG_LAYER_TAG)
            else:
                network = lasagne.layers.DenseLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf,
                    n_samples=self.n_samples)

        # Output layer
        if self.layers_type[len(self.n_hidden)] == 1:
            # Probabilistic layer (1) or deterministic layer (0).
            network = BNNNGLayer(
                network, self.n_out, num_data=self.n_batches,
                n_samples=self.n_samples, nonlinearity=self.outf,
                prior_sd=self.prior_sd, name=BNNNG_LAYER_TAG)
        else:
            network = lasagne.layers.DenseLayer(
                network, self.n_out, nonlinearity=self.outf,
                n_samples=self.n_samples)

        self.network = network

    def build_model(self):
        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        input_var = T.matrix('inputs',
                             dtype=theano.config.floatX)  # @UndefinedVariable
        target_var = T.matrix('targets',
                              dtype=theano.config.floatX)  # @UndefinedVariable

        # likelihood
        log_py_xw, pred = self.log_py_xw(input_var, target_var)
        # TODO: implement true fisher for s_grads
        activations, ss, ws = self.get_internel_results()
        w_grads = T.grad(T.sum(T.mean(log_py_xw, axis=1)), ws)
        if self.true_fisher:
            s_grads = T.grad(T.sum(pred)/self.likelihood_sd, ss)
        else:
            s_grads = T.grad(T.sum(log_py_xw), ss)
        layers = lasagne.layers.get_all_layers(self.network)[1:]
        updates = self.get_updates(layers, ws, w_grads, activations, s_grads)

        step_size = T.scalar('step_size', dtype=theano.config.floatX)  # @UndefinedVariable
        def fast_kl_div(log_py_xw, pred, layers, step_size):
            kl_component = []
            for l in layers:
                u = l.get_u()
                v = l.get_v()
                grad = T.grad(T.sum(T.mean(log_py_xw, axis=1)), l.W)
                grad = T.mean(grad, axis=0)
                kl = T.sum(grad * T.dot(u, T.dot(grad, v)))

                a = l.activation
                s = l.pre_activation
                if self.true_fisher:
                    s_grad = T.grad(T.sum(pred)/self.likelihood_sd, s)
                else:
                    s_grad = T.grad(T.sum(log_py_xw), s)
                # TODO: check if this approximation is reasonable
                a_t = a.dimshuffle((0, 2, 1))
                grad_u = T.mean(T.batched_dot(a_t, a), axis=0) / T.shape(a)[1]
                kl = kl + T.sum(grad_u * T.dot(u, T.dot(grad_u, u))) / (2 * l.num_units)

                s_grad_t = s_grad.dimshuffle((0, 2, 1))
                grad_v = T.mean(T.batched_dot(s_grad_t, s_grad), axis=0) / T.shape(s_grad)[1]
                kl = kl + T.sum(grad_v * T.dot(v, T.dot(grad_v, v))) / (2 * l.num_inputs)

                kl_component.append(T.square(step_size) * kl)
            return sum(kl_component)
        compute_fast_kl_div = fast_kl_div(log_py_xw, pred, layers, step_size)

        # Loss function
        loss = self.log_p_w_q_w_kl() / self.n_batches - T.mean(log_py_xw)
        if self.use_reverse_kl_reg:
            loss += self.reverse_kl_reg_factor * self.reverse_log_p_w_q_w_kl() / self.n_batches
        # Train/val fn.
        self.pred_fn = ext.compile_function(
            [input_var], self.pred_sym(input_var), log_name='pred_fn')
        self.train_fn = ext.compile_function(
            [input_var, target_var], loss, updates=updates, log_name='train_fn')
        self.train_update_fn = ext.compile_function(
            [input_var, target_var, step_size], compute_fast_kl_div, log_name='f_compute_fast_kl_div')

        # called kl div closed form but should be called surprise
        self.f_kl_div_closed_form = ext.compile_function(
            [], self.surprise(), log_name='kl_div_fn')

if __name__ == '__main__':
    pass
