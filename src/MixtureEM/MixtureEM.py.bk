#%%
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.experimental.numpy as tnp
from tqdm import tqdm
import numpy as np

tfd = tfp.distributions

class MixtureEM(tfp.distributions.Distribution):
    """
    Mixture model with EM algorithm for training.

    Parameters
    ----------
    initial_probs : array-like
        Initial probabilities of each component.
    components : list
        List of distributions for each component.
    fixed_weights : array-like, optional
        Array of booleans indicating which prior probabilities should be fixed. By default, all prior probabilities will be optimized.

    Attributes
    ----------
    logits : tf.Variable
        Log probabilities of each component.
    components : list
        List of distributions for each component.

    Methods
    -------
    fit(data, algo="EM", **kwargs)
        Fit the model to the data using the specified algorithm.
    
    get_mixture_distribution()
        Get a frozen version of the mixture distribution.

    sample(num_samples)
        Sample from the mixture distribution.

    posterior_probs(x)
        Compute the posterior probabilities of each component given the data.
         
    predict(x)
        Predict the most probable component for each data point.

    dump_pars(filename)
        Save the model parameters to a file.
    
    load_pars_from_file(filename)
        Load the model parameters from a file.

    Examples
    --------
    mu0 = tf.Variable([0.], dtype=tf.float32, name="mu0")
    sigma0 = tf.Variable([0.5], dtype=tf.float32, name="sigma0")

    mu1 = tf.Variable([10.], dtype=tf.float32, name="mu1")
    sigma1 = tf.Variable([0.5], dtype=tf.float32, name="sigma1")

    model0 = tfd.Normal(loc=mu0, scale=sigma0)
    model1 = tfd.Normal(loc=mu1, scale=sigma1)

    model = MixtureEM(initial_probs=[0.51, 0.49],
                    components=[model0, model1])

    optimizer_EM = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.fit(data, optimizer_EM, algo="EM",opt_steps=100, likelihood_opt_steps=500, lr=1e-3)

    """
    def __init__(
        self,
        initial_probs,
        components,
        fixed_weights=None,
        validate_args=False,
        allow_nan_stats=False,
        name="MixtureModelEM",
    ):
        super(MixtureEM, self).__init__(
            tf.float32,
            tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

        self.logits = tf.Variable(
            initial_value=tf.math.log(initial_probs), name="log_probs", trainable=True, dtype=tf.float32
        )
        self.components = components

        dist_variables = []
        for component in self.components:
            dist_variables.append(component.trainable_variables)
        self.dist_variables = tf.nest.flatten(dist_variables)

        self.num_components = len(self.components)

        self.fixed_weights = fixed_weights
        if self.fixed_weights is not None:
            self.weight_mask = tf.convert_to_tensor(fixed_weights, dtype=tf.bool)
        
        
        return
    
    @tf.function
    def opt_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_sum(self._log_prob(x)) # compute the negative log likelihood with current parameters (E pass)
        gradients = tape.gradient(loss, self.trainable_variables) # compute the gradients
        optimizer.apply_gradients(zip(gradients, self.trainable_variables)) # update the parameters (M pass)
        return loss

    
    def fit_GD(self, data, optimizer, max_steps=1000, verbose_level=1, rtol=1e-5, patience=20):
        opt_vars = self.trainable_variables
        optimizer.build(opt_vars)

        if verbose_level > 0:
            iterator = tqdm(np.arange(max_steps))
        else:
            iterator = np.arange(max_steps)

        x = tf.convert_to_tensor(data, dtype=tf.float32)

        last_loss = self.opt_step(x, optimizer)

        pat = 0 # patience counter

        for i in iterator:
            loss = self.opt_step(x, optimizer)
            if verbose_level > 0:
                iterator.set_postfix_str("Loss value: {:.5f}".format(loss.numpy()))
            if np.abs((last_loss - loss)/last_loss) < rtol and i>max_steps/40:
                if patience > pat:
                    pat += 1
                else:
                    break
            else:
                pat = 0

            last_loss = loss
        
        return

    def fit_EM(self, data, optimizer, opt_steps=10, likelihood_opt_steps=500, verbose_level=1, lr=1e-3):

        # self.optimizer.learning_rate.assign(lr)
        optimizer.build(self.dist_variables)

        data = tf.convert_to_tensor(data, dtype=tf.float32)
        if verbose_level > 0:
            iterator = tqdm(np.arange(opt_steps))
        else:
            iterator = np.arange(opt_steps)

        for step in iterator:
            # E-step: Calculate responsibilities
            responsibilities = tf.stop_gradient(
                self._compute_responsibilities(data)
            )  # make sure no gradietns flow through this step

            # M-step: Update weights and distributions
            if self.fixed_weights is not None:
                self._update_probs_fixed(responsibilities)
            else:
                self._update_probs(responsibilities)
            self._update_distributions(
                data, responsibilities, optimizer, likelihood_opt_steps, verbose=verbose_level > 1
            )
            loss = tf.reduce_sum(self.log_prob(data))
            if verbose_level > 0:
                iterator.set_postfix_str("Log Likelihood: {:.4f}".format(loss))
                # with np.printoptions(precision=4):
                #     iterator.set_postfix_str("Responsibilities: {}".format(responsibilities.numpy()[0]))
        return

    def fit(self, data, optimizer, algo="EM", **kwargs):
        if algo == "EM":
            self.fit_EM(data, optimizer, **kwargs)
        elif algo == "GD":
            self.fit_GD(data, optimizer, **kwargs)
        else:
            raise ValueError("Algorithm not recognized")
        return


    @tf.function
    def _compute_responsibilities(self, data):
        # log likelihood of each data point under each distribution
        # shape: (num_data, num_components)
        log_likelihoods = tf.stack(
            [component.log_prob(data) for component in self.components], axis=-1
        )
        # weighted probabilities of each data point under each distribution
        # shape: (num_data, num_components)
        weighted_log_likelihoods = log_likelihoods + self.logits
        # compute normalization
        # shape: (num_data,)
        log_normalization = tf.math.reduce_logsumexp(
            weighted_log_likelihoods, axis=-1, keepdims=True
        )
        # normalize to get responsibilities
        # shape: (num_data, num_components)
        responsibilities = tf.math.exp(weighted_log_likelihoods - log_normalization)
        return responsibilities

    @tf.function
    def _update_probs(self, responsibilities):
        probs = tf.reduce_mean(responsibilities, axis=0)
        logits = tf.math.log(probs)
        self.logits.assign(logits)
        return

    @tf.function
    def _update_probs_fixed(self, responsibilities):
        """
        Updates weights based on a fixed mask.
        """
        weights = tf.math.exp(self.logits)
        new_weights = tf.reduce_mean(responsibilities, axis=0)

        # Identify indices of weights to be updated
        update_idxs = tnp.ravel(tnp.nonzero(~self.weight_mask))

        free_idx = update_idxs[:-1]

        # Calculate the sum of fixed weights
        fixed_sum = tf.reduce_sum(weights[self.weight_mask])

        updates = tf.gather(new_weights, free_idx)
        # Calculate the weight for the conditional index
        updates = tf.concat([updates, [1 - fixed_sum - tf.reduce_sum(updates)]], axis=0)
        weights = tf.tensor_scatter_nd_update(weights, tf.reshape(update_idxs, (-1,1)), updates)

        logits = tf.math.log(weights)
        self.logits.assign(logits)
        return

    @tf.function
    def expectation_log_joint(self, data, responsibilities):
        components_log_prob = tf.stack(
            [component.log_prob(data) for component in self.components], axis=-1
        )
        arg = responsibilities * (tf.stop_gradient(self.logits) + components_log_prob)
        return tf.reduce_sum(arg)

    @tf.function
    def _update_distributions_step(self, data, responsibilities, optimizer, opt_steps=100):
        with tf.GradientTape() as tape:
            loss = -self.expectation_log_joint(data, responsibilities)
        grads = tape.gradient(loss, self.dist_variables)
        optimizer.apply_gradients(zip(grads, self.dist_variables))
        return loss

    def _update_distributions(
        self, data, responsibilities, optimizer, opt_steps=100, verbose=False
    ):
        if verbose:
            iterator = tqdm(np.arange(opt_steps))
        else:
            iterator = np.arange(opt_steps)

        for step in iterator:
            loss = self._update_distributions_step(data, responsibilities, optimizer, opt_steps)
            if verbose:
                iterator.set_postfix_str("Loss value: {:.5f}".format(loss))
        return loss

    def get_mixture_distribution(self):
        return tfd.Mixture(
            cat=tfd.Categorical(logits=self.logits), components=self.components
        )

    def get_trainable_variables(self):
        return self.get_mixture_distribution().trainable_variables

    def get_pars(self):
        vars = self.get_trainable_variables()
        dict_vars = {v.name: v.numpy() for v in vars}
        return dict_vars

    def load_pars(self, pars):
        for v in self.get_trainable_variables():
            v.assign(pars[v.name])
        return

    def dump_pars(self, filename):
        pars = self.get_pars()
        np.savez(filename, **pars)
        return

    def load_pars_from_file(self, filename):
        pars = np.load(filename)
        self.load_pars(pars)
        return
    
    def _get_mixture_dist(self):
        return tfd.Mixture(
            cat=tfd.Categorical(logits=self.logits), components=self.components
        )

    def sample(self, num_samples):
        return self._get_mixture_dist().sample(num_samples)

    def _log_prob(self, x):
        x = tf.convert_to_tensor(x, name="x")
        distribution_log_probs = [d.log_prob(x) for d in self.components]
        cat_log_probs = tf.math.log_softmax(self.logits)
        ndims = len(distribution_log_probs[0].shape)
        cat_target_shape = [len(self.components)] + [1] * ndims
        cat_log_probs = tf.reshape(cat_log_probs, cat_target_shape)
        final_log_probs = tf.add(cat_log_probs, distribution_log_probs)
        concat_log_probs = tf.stack(final_log_probs, 0)
        log_sum_exp = tf.reduce_logsumexp(concat_log_probs, axis=[0])
        return log_sum_exp

    def posterior_probs(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return self._compute_responsibilities(x)
    
    def predict(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.argmax(self.posterior_probs(x), axis=1)

    def _event_shape(self):
        return self.components[0].event_shape
        
    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return {}

    @property
    def probs_parameter(self):
        return tf.math.exp(self.logits)


# %%
