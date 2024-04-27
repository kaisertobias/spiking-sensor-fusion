import nengo
import numpy as np
import KalmanFilterProcesses as processes


class BaseKalmanFilter(object):
    """
    The BaseKalmanFilter class serves as a foundation for simulating Kalman-Filter algorithms within a Nengo neural
    simulation environment. It encapsulates the common properties and methods required to model and simulate the
    behavior of various Kalman filters with configurable parameters and neural network settings.

    Attributes
    ----------
    _dim_x : int
        Dimensionality of the state vector.
    _dim_z : int
        Dimensionality of the measurement vector.
    _dt : float
        Time step width.
    _dim_u : int
        Dimensionality of the control input vector. Default is 0.
    _n_neurons : int
        Number of neurons per dimension in the state and covariance representation. Default is 100.
    _measure_func : callable
        Function to simulate the measurement. Should take time as input and return measurement vector.
    _control_input_func : callable
        Function to simulate the control input. Should take time as input and return control input vector.
    _initial_x : numpy.ndarray
        Initial state vector.
    _initial_P : numpy.ndarray
        Initial covariance matrix.
    _connection_synapse : float
        Synapse parameter for Nengo connections. Default is 0.1.
    _x_radius : float
        Radius of the state representation ensembles. Default is 1.
    _P_radius : float
        Radius of the covariance representation ensembles. Default is 1.
    _x_tau_rc : float
        Membrane RC time constant of the state representation ensembles. Default is 0.02 seconds.
    _P_tau_rc : float
        Membrane RC time constant of the covariance representation ensembles. Default is 0.02 seconds.
    _x_tau_ref : float
        Absolute refractory period of the state representation ensembles in seconds. Default is 0.002 seconds.
    _P_tau_ref : float
        Absolute refractory period of the covariance representation ensembles. Default is 0.002 seconds.
    _x_max_rates, _P_max_rates : nengo.dists.Distribution or numpy.ndarray
        Maximum firing rates for neurons representing state and covariance.
    _x_intercepts, _P_intercepts : nengo.dists.Distribution or numpy.ndarray
        Intercepts for neurons representing state and covariance.
    _x_encoders, _P_encoders : nengo.dists.Distribution or numpy.ndarray
        Encoder values for neurons representing state and covariance.
    _sample_rate: float
        The rate in which probes are taken from the nengo simulation. Defautl is dt.
    _simulation_runtime : float
        Total runtime for the simulation. Default is 1 second.
    _simulation_step_size : float
        Simulation step size. Default is 0.001 seconds.
    _model : nengo.Network
        The Nengo network model.
    _simulator : nengo.Simulator
        The Nengo simulator.
    _x_posterior, _x_prior : nengo.networks.EnsembleArray
        Ensembles representing the state vector.
    _P_posterior, _P_prior : nengo.networks.EnsembleArray
        Ensembles representing the covariance matrix.
    _predict, _update : nengo.Process
        Processes for the predict and update steps of the Kalman filter.

    Methods
    -------
    build_x_ensemble(), build_P_ensemble()    
        Methods to build the state and covariance representation ensembles for user-defined parametrization.
    build_model()
        Constructs the Nengo model for the Kalman-Filter simulation.
    build_model_for_measurements(zs)
        Build the Nengo model for the Kalman-Filter simulation for a dataset of simulation data 'zs'.
        Parameter 'measure_func' is overwritten according to 'zs'.
    execute()
        Runs the Nengo simulator with the constructed model.
    get_probes()
        Retrieves data from the probes after simulation.
    """

    def __init__(self, dim_x, dim_z, dt):
        """
        Parameters
        ----------
        dim_x : int
            Dimensionality of the state vector.
        dim_z : int
            Dimensionality of the measurement vector.
        dt : float
            Time step for the simulation.
        """
        self._dim_x = dim_x
        self._P_size = int(dim_x *(dim_x + 1) / 2)
        self._dim_z = dim_z
        self._dim_u = 0
        self._dt = dt
        self._n_neurons = 100
        self._measure_func = None
        self._control_input_func = None
        self._initial_x = None
        self._initial_P = None
        self._connection_synapse = .1
        self._x_radius = 1
        self._P_radius = 1
        self._x_tau_rc = .02
        self._P_tau_rc = .02
        self._x_tau_ref = .002
        self._P_tau_ref = .002
        self._x_max_rates = nengo.dists.Uniform(200, 400)
        self._P_max_rates = nengo.dists.Uniform(200, 400)
        self._x_intercepts = nengo.dists.Uniform(-1.0, 0.9)
        self._P_intercepts = nengo.dists.Uniform(-1.0, 0.9)
        self._x_encoders = nengo.dists.ScatteredHypersphere(surface=True)
        self._P_encoders = nengo.dists.ScatteredHypersphere(surface=True)
        self._sample_rate = dt
        self._simulation_runtime = 1.
        self._simulation_step_size = .001

        self._model = None
        self._simulator = None
        self._x_posterior = None
        self._x_prior = None
        self._P_posterior = None
        self._P_prior = None
        self._init_x_node = None
        self._init_P_node = None
        self._u_node = None
        self._measurement_node = None
        self._predict = None
        self._predict_node = None
        self._update = None
        self._update_node = None
        self._x_probe = None
        self._P_probe = None

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
    def initial_P(self):
        return self._initial_P

    @initial_P.setter
    def initial_P(self, value):
        self._initial_P = value

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
    def P_radius(self):
        return self._P_radius

    @P_radius.setter
    def P_radius(self, value):
        if value < 0:
            raise ValueError('Radius of P under 0 is not possible')
        self._P_radius = value

    @property
    def x_tau_rc(self):
        return self._x_tau_rc

    @x_tau_rc.setter
    def x_tau_rc(self, value):
        if value <= 0:
            raise ValueError('tau_rc of x-Ensemble less or equal to 0 is not possible')
        self._x_tau_rc = value
    
    @property
    def P_tau_rc(self):
        return self._P_tau_rc

    @P_tau_rc.setter
    def P_tau_rc(self, value):
        if value <= 0:
            raise ValueError('tau_rc of P-Ensemble less or equal to 0 is not possible')
        self._P_tau_rc = value

    @property
    def x_tau_ref(self):
        return self._x_tau_ref

    @x_tau_ref.setter
    def x_tau_ref(self, value):
        if value <= 0:
            raise ValueError('tau_ref of x-Ensemble less or equal to 0 is not possible')
        self._x_tau_ref = value
    
    @property
    def P_tau_ref(self):
        return self._P_tau_ref

    @P_tau_ref.setter
    def P_tau_ref(self, value):
        if value <= 0:
            raise ValueError('tau_ref of P-Ensemble less or equal to 0 is not possible')
        self._P_tau_ref = value

    @property
    def x_intercepts(self):
        return self._x_intercepts

    @x_intercepts.setter
    def x_intercepts(self, value):
        self._x_intercepts = value
    
    @property
    def P_intercepts(self):
        return self._P_intercepts

    @P_intercepts.setter
    def P_intercepts(self, value):
        self._P_intercepts = value
    
    @property
    def x_encoders(self):
        return self._x_encoders

    @x_encoders.setter
    def x_encoders(self, value):
        self._x_encoders = value
    
    @property
    def P_encoders(self):
        return self._P_encoders

    @P_encoders.setter
    def P_encoders(self, value):
        self._P_encoders = value

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
    
    def build_P_ensemble(self):
        """
        Build a covariance representation ensemble.
        """
        ensemble = nengo.networks.EnsembleArray(
            n_neurons=self._n_neurons, 
            n_ensembles=self._P_size,
            radius=self._P_radius, 
            neuron_type=nengo.LIF(tau_rc=self._P_tau_rc, tau_ref=self._P_tau_ref),
            max_rates=self._P_max_rates,
            intercepts=self._P_intercepts,
            encoders=self._P_encoders
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
            self._P_posterior = self.build_P_ensemble()
            self._P_prior = self.build_P_ensemble()

            #Initialzustände zuweisen
            self._init_x_node = nengo.Node(lambda t: self._initial_x if t <= self._dt else np.zeros(self._dim_x))
            nengo.Connection(self._init_x_node, self._x_posterior.input)
            self._init_P_node = nengo.Node(lambda t: self._initial_P if t <= self._dt else np.zeros(self._P_size))
            nengo.Connection(self._init_P_node, self._P_posterior.input)

            #Control-Input eines Benutzers
            self._u_node = nengo.Node(self.control_input_func, size_out=self._dim_u)

            #Simmulation einer Messung über Zeit
            self._measurement_node = nengo.Node(self._measure_func, size_out=self._dim_z)

            #Predict-Step
            self._predict_node = nengo.Node(self._predict, size_in=self._dim_x+self._P_size+self._dim_u, size_out=self._dim_x+self._P_size)
            nengo.Connection(self._x_posterior.output, self._predict_node[:self._dim_x], synapse=self._connection_synapse)
            nengo.Connection(self._P_posterior.output, self._predict_node[self._dim_x:self._dim_x+self._P_size], synapse=self._connection_synapse)
            if self._dim_u > 0:
                nengo.Connection(self._u_node, self._predict_node[self._dim_x+self._P_size:], synapse=self._connection_synapse)
            nengo.Connection(self._predict_node[:self._dim_x], self._x_prior.input, synapse=self._connection_synapse)
            nengo.Connection(self._predict_node[self._dim_x:], self._P_prior.input, synapse=self._connection_synapse)

            #Update-Step
            self._update_node = nengo.Node(self._update, size_in=self._dim_x+self._P_size+self._dim_z, size_out=self._dim_x+self._P_size)
            nengo.Connection(self._x_prior.output, self._update_node[:self._dim_x], synapse=self._connection_synapse)
            nengo.Connection(self._P_prior.output, self._update_node[self._dim_x:self._dim_x+self._P_size], synapse=self._connection_synapse)
            nengo.Connection(self._measurement_node, self._update_node[self._dim_x+self._P_size:], synapse=self._connection_synapse)
            nengo.Connection(self._update_node[:self._dim_x], self._x_posterior.input, synapse=self._connection_synapse)
            nengo.Connection(self._update_node[self._dim_x:], self._P_posterior.input, synapse=self._connection_synapse)

            #Probe des Ortes, der Geschwindigkeit und aller (Co-)Varianzen
            self._x_probe = nengo.Probe(self._x_posterior.output, sample_every=self._sample_rate, synapse=self._connection_synapse)
            self._P_probe = nengo.Probe(self._P_posterior.output, sample_every=self._sample_rate, synapse=self._connection_synapse)
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
        if self._P_probe != None:
            probes['P'] = self._simulator.data[self._P_probe]
        return probes



class LinearKF(BaseKalmanFilter):
    """
    The LinearKF class implements a linear Kalman-Filter within a Nengo neural simulation environment. It extends
    the BaseKalmanFilter class with specific predict and update processes suited for linear dynamics and measurement
    models.

    Attributes
    ----------
    _F : numpy.ndarray
        State transition matrix.
    _Q : numpy.ndarray
        Process noise covariance matrix.
    _H : numpy.ndarray
        Measurement matrix.
    _R : numpy.ndarray
        Measurement noise covariance matrix.
    _B : numpy.ndarray
        Control input matrix.
    """

    def __init__(self, dim_x, dim_z, dt, F, Q, H, R, B=None):
        """
        Parameters
        ----------
        dim_x : int
            Dimensionality of the state vector.
        dim_z : int
            Dimensionality of the measurement vector.
        dt : float
            Time step for the simulation.
        F : numpy.ndarray
            State transition matrix.
        Q : numpy.ndarray
            Process noise covariance matrix.
        H : numpy.ndarray
            Measurement matrix.
        R : numpy.ndarray
            Measurement noise covariance matrix.
        B : numpy.ndarray, optional
            Control input matrix.
        """
        super().__init__(dim_x, dim_z, dt)
        self._predict = processes.LinearKF_Predict(dim_x=dim_x, F=F, Q=Q, B=B, dim_u=self.dim_u)
        self._update = processes.LinearKF_Update(dim_x=dim_x, dim_z=dim_z, H=H, R=R)



class EKF(BaseKalmanFilter):
    """
    The EKF class implements an Extended Kalman Filter (EKF) within a Nengo neural simulation environment. It utilizes
    nonlinear functions for the state transition and measurement processes, making it suitable for systems where the
    linear approximation is insufficient.

    Attributes
    ----------
    _F_at : callable
            Nonlinear function that computes matrix F for current state estimate.
    _Q : numpy.ndarray
        Process noise covariance matrix.
    _h : callable
        Nonlinear measurement function.
    _H_Jacobian : callable
        Function to compute the Jacobian of the measurement function.
    _R : numpy.ndarray
        Measurement noise covariance matrix.
    """

    def __init__(self, dim_x, dim_z, dt, f, F_Jacobian, Q, h, H_Jacobian, R):
        """
        Parameters
        ----------
        dim_x : int
            Dimensionality of the state vector.
        dim_z : int
            Dimensionality of the measurement vector.
        dt : float
            Time step for the simulation.
        F_at : callable
            Nonlinear function that computes matrix F for current state estimate.
        Q : numpy.ndarray
            Process noise covariance matrix.
        h : callable
            Nonlinear measurement function.
        H_Jacobian : callable
            Function to compute the Jacobian of the measurement function.
        R : numpy.ndarray
            Measurement noise covariance matrix.
        """
        super().__init__(dim_x, dim_z, dt)
        self._predict = processes.EKF_Predict(dim_x=dim_x, f=f, F_Jacobian=F_Jacobian, Q=Q, dim_u=self._dim_u)
        self._update = processes.EKF_Update(dim_x=dim_x, dim_z=dim_z, h=h, H_Jacobian=H_Jacobian, R=R)



class UKF(BaseKalmanFilter):
    """
    The UKF class implements an Unscented Kalman Filter (UKF) within a Nengo neural simulation environment. The UKF
    is designed for systems with nonlinear dynamics and measurements, utilizing the unscented transform to more
    accurately capture the state's probability distribution.

    Attributes
    ----------
    _f : callable
        Nonlinear state transition function.
    _Q : numpy.ndarray
        Process noise covariance matrix.
    _h : callable
        Nonlinear measurement function.
    _R : numpy.ndarray
        Measurement noise covariance matrix.
    _kappa : float
        Tuning parameter for the unscented transform. Default is 1.
    
    Methods
    -------
    def compute_weights()
        Compute weights for unscented transform.
    """
     
    def __init__(self, dim_x, dim_z, dt, f, Q, h, R, kappa=1.):
        super().__init__(dim_x, dim_z, dt)
        """
        Parameters
        ----------
        dim_x : int
            Dimensionality of the state vector.
        dim_z : int
            Dimensionality of the measurement vector.
        dt : float
            Time step for the simulation.
        f : callable
            Nonlinear state transition function.
        Q : numpy.ndarray
            Process noise covariance matrix.
        h : callable
            Nonlinear measurement function.
        R : numpy.ndarray
            Measurement noise covariance matrix.
        kappa : float, optional
            Tuning parameter for the unscented transform. Default is 1.
        """
        self._f = f
        self._Q = Q
        self._h = h
        self._R = R
        self._kappa = kappa
        self._W = self.compute_weights()
        self._sigmas = np.zeros((2*dim_x+1, dim_x)) 
        self._sigmas_f = np.zeros((2*dim_x+1, dim_x)) 
        self._sigmas_h = np.zeros((2*dim_x+1, dim_z))
        self._predict = processes.UKF_Predict(dim_x=dim_x, W=self._W, kappa=self._kappa, sigmas=self._sigmas, sigmas_f=self._sigmas_f, f=f, Q=Q, dim_u=self.dim_u)
        self._update = processes.UKF_Update(dim_x=dim_x, dim_z=dim_z, W=self._W, sigmas_f=self._sigmas_f, sigmas_h=self._sigmas_h, h=h, R=R)

    @property
    def kappa(self):
        return self.kapppa

    @kappa.setter
    def kappa(self, value):
        self._kappa = value
        self._W = self.compute_weights()
        self.predict = processes.UKF_Predict(dim_x=self._dim_x, W=self._W, kappa=self._kappa, sigmas=self._sigmas, sigmas_f=self._sigmas_f, f=self._f, Q=self._Q, dim_u=self._dim_u)
        self.update = processes.UKF_Update(dim_x=self._dim_x, dim_z=self._dim_z, W=self._W, sigmas_f=self._sigmas_f, sigmas_h=self._sigmas_h, h=self._h, R=self._R)
    
    @property
    def W(self):
        return self._W

    def compute_weights(self):
        W = np.full(2*self._dim_x+1, 1. / 2*(self._dim_x + self._kappa))
        W[0] = self._kappa / (self._dim_x + self._kappa)
        return W

