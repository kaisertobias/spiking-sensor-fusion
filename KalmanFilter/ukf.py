import nengo
import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky


class BaseProcess(nengo.Process):
    def __init__(self, dim_x, W):
        super(BaseProcess, self).__init__()
        self._dim_x = dim_x
        self._W = W
    
    def compute_mean_and_covariance(self, sigmas, noise):
        #Berechnung des Mittelwerts der Sigma-Punkte
        mean = self._W @ sigmas
        #Berechnung des residuals
        y = sigmas - mean[np.newaxis, :]
        cov = y.T @ np.diag(self._W) @ y + noise
        return (mean, cov)


    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        raise NotImplementedError
    


class UKFPredict(BaseProcess):
    def __init__(self, dim_x, P, W, kappa, sigmas, sigmas_f, f, Q, dim_u=0):
        super().__init__(dim_x=dim_x, W=W)
        self._P = P
        self._kappa = kappa
        self._sigmas = sigmas
        self._sigmas_f = sigmas_f
        self._f = f
        self._Q = Q
        self._dim_u = dim_u

    def from_model_space(self, data):
        x = np.asarray(data[:self._dim_x]).reshape((self._dim_x,))
        u = np.asarray(data[self._dim_x:]).reshape((self._dim_u,))
        return (x, u)

    def eigenvalue_decomposition(self):
        eigenvalues, U = np.linalg.eigh((self._dim_x + self._kappa) * self._P)
        lambda_prime = np.diag(np.maximum(eigenvalues, 0))
        lambda_prime_sqrt = np.sqrt(lambda_prime)
        _, R = np.linalg.qr((U @ lambda_prime_sqrt).T)
        return R

    def compute_cholesky_decomposition(self):
        try:
            U = cholesky((self._dim_x + self._kappa) * self._P)
        except np.linalg.LinAlgError:
            U = self.eigenvalue_decomposition()
        return U
    
    def compute_sigmas(self, x):
        U = self.compute_cholesky_decomposition()
        sigmas = np.zeros((2*self._dim_x+1, self._dim_x))
        sigmas[0] = x
        for i in range(self._dim_x):
            if np.isinf(U[i]).any():
                print(U[i])
            sigmas[i+1] = x + U[i]
            sigmas[self._dim_x+i+1] = x - U[i]
        return sigmas
        
    def compute_f_of_sigmas(self, u, t):
        for i, s in enumerate(self._sigmas):
            self._sigmas_f[i] = self._f(s, t)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, data):
            x, u = self.from_model_space(data)
            
            self._sigmas = self.compute_sigmas(x)
            self.compute_f_of_sigmas(u, t)
            x, self._P = self.compute_mean_and_covariance(self._sigmas_f, self._Q)
            self._sigmas_f = self.compute_sigmas(x)

            return x.flatten()
        return step



class UKFUpdate(BaseProcess):
    def __init__(self, dim_x, dim_z, P, W, sigmas_f, sigmas_h, h, R):
        super().__init__(dim_x=dim_x, W=W)
        self._dim_z = dim_z
        self._P = P
        self._sigmas_f = sigmas_f
        self._sigmas_h = sigmas_h
        self._h = h
        self._R = R

    def from_model_space(self, data):
        x = np.asarray(data[:self._dim_x]).reshape((self._dim_x,))
        z = np.asarray(data[self._dim_x:]).reshape((self._dim_z,))
        return (x, z)

    def compute_h_of_sigmas(self):
        for i, s in enumerate(self._sigmas_f):
            self._sigmas_h[i] = self._h(s)

    def compute_cross_variance(self, x, zp):
        Pxz = np.zeros((self._dim_x, self._dim_z))
        for i in range(2*self._dim_x+1):
            dx = np.subtract(self._sigmas_f[i], x)
            dz = np.subtract(self._sigmas_h[i], zp)
            Pxz += self._W[i] * np.outer(dx, dz)
        return Pxz
    
    def compute_klaman_gain(self, Pxz, S):
        S_inv = inv(S)
        return Pxz @ S_inv
    
    def update_state(self, x, K, z, zp):
        y = np.subtract(z, zp)
        return x + K @ y
    
    def update_cov_matrix(self, K, S):
        return self._P - K @ S @ K.T

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, data):
            x, z = self.from_model_space(data)
            
            #Sigma-Punkte in Betrachtungsraum bringen
            self.compute_h_of_sigmas()

            #Durchschnitt und Kovarianz der Prediction
            zp, S = self.compute_mean_and_covariance(self._sigmas_h, self._R)
            
            #Cross-Variance des Status und der Messung
            Pxz = self.compute_cross_variance(x, zp)

            #Kalman Gain
            K = self.compute_klaman_gain(Pxz, S)

            #Update state and covariance
            x = self.update_state(x, K, z, zp)
            self._P = self.update_cov_matrix(K, S)
            
            return x.flatten()
        return step



class UnscentedKalmanFilter(object):
    def __init__(self, dim_x, dim_z, dt, P, f, Q, h, R, kappa=1.):
        self._dim_x = dim_x
        self._dim_z = dim_z
        self._dim_u = 0
        self._dt = dt
        self._P = P
        self._f = f
        self._Q = Q
        self._h = h
        self._R = R
        self._kappa = kappa
        self._W = self.compute_weights()
        self._sigmas = np.zeros((2*dim_x+1, dim_x)) 
        self._sigmas_f = np.zeros((2*dim_x+1, dim_x)) 
        self._sigmas_h = np.zeros((2*dim_x+1, dim_z))
        self._predict = UKFPredict(dim_x=dim_x, W=self._W, P=self._P, kappa=self._kappa, sigmas=self._sigmas, sigmas_f=self._sigmas_f, f=f, Q=Q, dim_u=self.dim_u)
        self._update = UKFUpdate(dim_x=dim_x, dim_z=dim_z, W=self._W, P=self._P, sigmas_f=self._sigmas_f, sigmas_h=self._sigmas_h, h=h, R=R)
        self._n_neurons = 100
        self._measure_func = None
        self._control_input_func = None
        self._initial_x = None
        self._connection_synapse = .1
        self._x_radius = 1
        self._x_tau_rc = .02
        self._x_tau_ref = .002
        self._x_max_rates = nengo.dists.Uniform(200, 400)
        self._x_intercepts = nengo.dists.Uniform(-1.0, 0.9)
        self._x_encoders = nengo.dists.ScatteredHypersphere(surface=True)
        self._sample_rate = dt
        self._simulation_runtime = 1.
        self._simulation_step_size = .001

        self._model = None
        self._simulator = None
        self._x_posterior = None
        self._x_prior = None
        self._init_x_node = None
        self._u_node = None
        self._measurement_node = None
        self._predict_node = None
        self._update_node = None
        self._x_probe = None

    @property
    def dim_x(self):
        return self._dim_x

    @dim_x.setter
    def dim_x(self, value):
        if value <= 0:
            raise ValueError('Dimension for the state estimate less or equal than 0 is not possible')
        self._dim_x = value
    
    @property
    def dim_z(self):
        return self._dim_z

    @dim_z.setter
    def dim_z(self, value):
        if value <= 0:
            raise ValueError('Dimension for the measurement less or equal than 0 is not possible')
        self._dim_z = value

    @property
    def dim_u(self):
        return self._dim_u

    @dim_u.setter
    def dim_u(self, value):
        if value < 0:
            raise ValueError('Dimension for the user input less than 0 is not possible')
        self._dim_u = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if value <= 0:
            raise ValueError('Value for dt less or equal than 0 is not possible')
        self._dt = value

    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, value):
        self._P = value

    @property
    def n_neurons(self):
        return self._n_neurons

    @n_neurons.setter
    def n_neurons(self, value):
        if value <= 0:
            raise ValueError('Number of neurons under 1 is not possible')
        self._n_neurons = value

    @property
    def measure_func(self):
        return self._measure_func

    @measure_func.setter
    def measure_func(self, value):
        self._measure_func = value

    @property
    def control_input_func(self):
        return self._control_input_func

    @control_input_func.setter
    def control_input_func(self, value):
        self._control_input_func = value

    @property
    def initial_x(self):
        return self._initial_x

    @initial_x.setter
    def initial_x(self, value):
        self._initial_x = value

    @property
    def connection_synapse(self):
        return self._connection_synapse

    @connection_synapse.setter
    def connection_synapse(self, value):
        if value < 0:
            raise ValueError('Connection-Synapse under 0 is not possible')
        self._connection_synapse = value
    
    @property
    def x_radius(self):
        return self._x_radius

    @x_radius.setter
    def x_radius(self, value):
        if value < 0:
            raise ValueError('Radius of x under 0 is not possible')
        self._x_radius = value
    
    @property
    def x_tau_rc(self):
        return self._x_tau_rc

    @x_tau_rc.setter
    def x_tau_rc(self, value):
        if value <= 0:
            raise ValueError('tau_rc of x-Ensemble less or equal to 0 is not possible')
        self._x_tau_rc = value

    @property
    def x_tau_ref(self):
        return self._x_tau_ref

    @x_tau_ref.setter
    def x_tau_ref(self, value):
        if value <= 0:
            raise ValueError('tau_ref of x-Ensemble less or equal to 0 is not possible')
        self._x_tau_ref = value

    @property
    def x_intercepts(self):
        return self._x_intercepts

    @x_intercepts.setter
    def x_intercepts(self, value):
        self._x_intercepts = value
    
    @property
    def x_encoders(self):
        return self._x_encoders

    @x_encoders.setter
    def x_encoders(self, value):
        self._x_encoders = value
    
    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if value < 0:
            raise ValueError('A smaple rate less or equal to 0s is not possible')
        self._sample_rate = value

    @property
    def simulation_runtime(self):
        return self._simulation_runtime

    @simulation_runtime.setter
    def simulation_runtime(self, value):
        if value < 0:
            raise ValueError('A runtime less or equal to 0s is not possible')
        self._simulation_runtime = value

    @property
    def simulation_step_size(self):
        return self._simulation_step_size

    @simulation_step_size.setter
    def simulation_step_size(self, value):
        if value < 0:
            raise ValueError('A simulation step size less or equal to 0s is not possible')
        self._simulation_step_size = value

    @property
    def predict(self):
        return self._predict

    @predict.setter
    def predict(self, value):
        self._predict = value

    @property
    def update(self):
        return self._update

    @update.setter
    def update(self, value):
        self._update = value

    @property
    def kappa(self):
        return self.kapppa

    @kappa.setter
    def kappa(self, value):
        self._kappa = value
        self._W = self.compute_weights()
        self.predict = UKFPredict(dim_x=self._dim_x, W=self._W, P=self._P, kappa=self._kappa, sigmas=self._sigmas, sigmas_f=self._sigmas_f, f=self._f, Q=self._Q, dim_u=self._dim_u)
        self.update = UKFUpdate(dim_x=self._dim_x, dim_z=self._dim_z, W=self._W, P=self._P, sigmas_f=self._sigmas_f, sigmas_h=self._sigmas_h, h=self._h, R=self._R)
    
    @property
    def W(self):
        return self._W

    def compute_weights(self):
        W = np.full(2*self._dim_x+1, 1. / 2*(self._dim_x + self._kappa))
        W[0] = self._kappa / (self._dim_x + self._kappa)
        return W    

    def build_x_ensemble(self):
        """
        Build a state representation ensemble.
        """
        ensemble = nengo.networks.EnsembleArray(
            n_neurons=self._n_neurons, 
            n_ensembles=self._dim_x,
            radius=self._x_radius, 
            neuron_type=nengo.LIF(tau_rc=self._x_tau_rc, tau_ref=self._x_tau_ref),
            max_rates=self._x_max_rates,
            intercepts=self._x_intercepts,
            encoders=self._x_encoders
        )
        return ensemble

    def build_model(self):
        """
        Build the Nengo model for the Kalman-Filter simulation.
        """
        self._model = nengo.Network()
        with self._model:
            #Darstellung des Zustandes und der Cov-Matrix durch Ensemble
            self._x_posterior = self.build_x_ensemble()
            self._x_prior = self.build_x_ensemble()

            #Initialzustände zuweisen
            self._init_x_node = nengo.Node(lambda t: self._initial_x if t <= self._dt else np.zeros(self._dim_x))
            nengo.Connection(self._init_x_node, self._x_posterior.input)

            #Control-Input eines Benutzers
            self._u_node = nengo.Node(self.control_input_func, size_out=self._dim_u)

            #Simmulation einer Messung über Zeit
            self._measurement_node = nengo.Node(self._measure_func, size_out=self._dim_z)

            #Predict-Step
            self._predict_node = nengo.Node(self._predict, size_in=self._dim_x+self._dim_u, size_out=self._dim_x)
            nengo.Connection(self._x_posterior.output, self._predict_node[:self._dim_x], synapse=self._connection_synapse)
            if self._dim_u > 0:
                nengo.Connection(self._u_node, self._predict_node[self._dim_x:], synapse=self._connection_synapse)
            nengo.Connection(self._predict_node, self._x_prior.input, synapse=self._connection_synapse)

            #Update-Step
            self._update_node = nengo.Node(self._update, size_in=self._dim_x+self._dim_z, size_out=self._dim_x)
            nengo.Connection(self._x_prior.output, self._update_node[:self._dim_x], synapse=self._connection_synapse)
            nengo.Connection(self._measurement_node, self._update_node[self._dim_x:], synapse=self._connection_synapse)
            nengo.Connection(self._update_node, self._x_posterior.input, synapse=self._connection_synapse)

            #Probe des Ortes, der Geschwindigkeit und aller (Co-)Varianzen
            self._x_probe = nengo.Probe(self._x_posterior.output, sample_every=self._sample_rate, synapse=self._connection_synapse)
        return self
    
    def build_model_for_measurements(self, zs):
        """
        Build the Nengo model for the Kalman-Filter simulation for a dataset of simulation data 'zs'.
        Parameter 'measure_func' is overwritten according to 'zs'.
        """
        if len(zs) == 0:
            raise AssertionError("Length of 'zs' is zero")
        if len(zs[0]) != self._dim_z:
            raise AssertionError("Dimension of an entry of 'zs' does not match 'dim_z'")

        def measure(t):
            measure.epoch += 1

            if measure.epoch == int(self._dt / self._simulation_step_size):
                measure.z_index += 1
                measure.epoch = 0

            if measure.z_index < 0:
                return np.zeros(measure.zs[0].shape)
            elif measure.z_index >= 0 and measure.z_index < len(measure.zs):
                return measure.zs[measure.z_index]
            else:
                return measure.zs[-1]
        measure.epoch = 0
        measure.z_index = -1
        measure.zs = zs

        self._measure_func = measure
        return self.build_model()

    def execute(self):
        """
        Runs the Nengo simulator with the constructed model.
        """
        try:
            self._simulator = nengo.Simulator(self._model, dt=self._simulation_step_size)
            self._simulator.run(self._simulation_runtime)
            return self
        except Exception as e:
            raise RuntimeError(f'Error while executing Kalman-Filter: {e}')

    def get_probes(self):
        """
        Retrieves data from the probes after a simulation.
        """
        probes = {}
        if self._simulator != None:
            probes['time_range'] = self._simulator.trange(sample_every=0.01)
        if self._x_probe != None:
            probes['x'] = self._simulator.data[self._x_probe]
        return probes

