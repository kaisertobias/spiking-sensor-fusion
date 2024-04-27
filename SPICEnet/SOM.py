import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
import nengo
import SOMProcesses as processes



class SOMNetwork(object):
    def __init__(self, dt=0.001, n_neurons=100, n_lif_per_neuron=100, n_xmod_learning_epochs=100, n_inner_learning_epochs=100, eta=1., xi=1e-3, 
                 sigma_def=.045, radius=1.5, tau_rc=.02, tau_ref=.002, sigma0=None, sigmaf=1., sigma_learning_type='exp', alpha0=.1, alphaf=.001, 
                 alpha_learning_type='exp'):
        self._learning_tpyes = ('sigmoid', 'invtime', 'exp')
        assert sigma_learning_type in self._learning_tpyes, "sigma_learning_type must be 'sigmoid', 'invtime' or 'exp'"
        assert alpha_learning_type in self._learning_tpyes, "alpha_learning_type must be 'sigmoid', 'invtime' or 'exp'"
        
        self._dt = dt
        self._n_som = 2
        self._n_neurons = n_neurons
        self._n_lif_per_neuron = n_lif_per_neuron
        self._n_xmod_learning_epochs = n_xmod_learning_epochs
        self._n_inner_learning_epochs = n_inner_learning_epochs
        self._eta = eta
        self._xi = xi
        self._sigma_def = sigma_def
        self._radius = radius
        self._tau_rc = tau_rc
        self._tau_ref = tau_ref
        if sigma0 is None:
            self._sigma0 = n_neurons / 2.
        else:
            self._sigma0 = sigma0
        self._sigmaf = sigmaf
        self._sigma_learning_type = sigma_learning_type
        self._alpha0 = alpha0
        self._alphaf = alphaf
        self._alpha_learning_type = alpha_learning_type

        self._sigma = self.parametrize_learning_law(self._sigma0 , sigmaf, 1, n_xmod_learning_epochs, sigma_learning_type)
        self._alpha = self.parametrize_learning_law(alpha0, alphaf, 1, n_xmod_learning_epochs, alpha_learning_type)

        self._model = None
        self._winput_x = None
        self._winput_y = None
        self._activity_x = None
        self._activity_y = None
        self._std_x = None
        self._std_y = None
        self._learn_node = None
        self._p_winput_x = None
        self._p_winput_y = None
        self._p_activity_x = None
        self._p_activity_y = None
        self._p_std_x = None
        self._p_std_y = None
        self._simulator = None

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if value <= 0:
            raise ValueError('Simulation step width less or equal than 0 is not possible')
        self._dt = value

    @property
    def n_neurons(self):
        return self._n_neurons

    @n_neurons.setter
    def n_neurons(self, value):
        if value <= 0:
            raise ValueError('Number of neurons less or equal than 0 is not possible')
        self._n_neurons = value

    @property
    def n_lif_per_neuron(self):
        return self._n_lif_per_neuron

    @n_lif_per_neuron.setter
    def n_lif_per_neuron(self, value):
        if value <= 0:
            raise ValueError('Number of LIF-neurons less or equal than 0 is not possible')
        self._n_lif_per_neuron = value

    @property
    def n_xmod_learning_epochs(self):
        return self._n_xmod_learning_epochs

    @n_xmod_learning_epochs.setter
    def n_xmod_learning_epochs(self, value):
        if value <= 0:
            raise ValueError('Number of learning epochs less or equal than 0 is not possible')
        self._n_xmod_learning_epochs = value
        self._sigma = self.parametrize_learning_law(self.sigma0 , self.sigmaf, 1, self._n_xmod_learning_epochs, self.sigma_learning_type)
        self._alpha = self.parametrize_learning_law(self.alpha0 , self.alphaf, 1, self._n_xmod_learning_epochs, self.alpha_learning_type)

    @property
    def n_inner_learning_epochs(self):
        return self._n_inner_learning_epochs

    @n_inner_learning_epochs.setter
    def n_inner_learning_epochs(self, value):
        if value <= 0:
            raise ValueError('Number of learning epochs less or equal than 0 is not possible')
        self._n_inner_learning_epochs = value

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, value):
        self._xi = value

    @property
    def sigma_def(self):
        return self._sigma_def

    @sigma_def.setter
    def sigma_def(self, value):
        self._sigma_def = value

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise ValueError('Radius cannot be less or equal to 0')
        self._radius = value

    @property
    def tau_rc(self):
        return self._tau_rc

    @tau_rc.setter
    def tau_rc(self, value):
        if value <= 0:
            raise ValueError('tau_rc cannot be less or equal to 0')
        self._tau_rc = value

    @property
    def tau_ref(self):
        return self._tau_ref

    @tau_ref.setter
    def tau_ref(self, value):
        if value <= 0:
            raise ValueError('tau_ref cannot be less or equal to 0')
        self._tau_ref = value

    @property
    def sigma0(self):
        return self._sigma0

    @sigma0.setter
    def sigma0(self, value):
        if value <= 0:
            raise ValueError('Sigma0 cannot be less or equal to 0')
        self._sigma0 = value
        self._sigma = self.parametrize_learning_law(self._sigma0 , self.sigmaf, 1, self.n_xmod_learning_epochs, self.sigma_learning_type)

    @property
    def sigmaf(self):
        return self._sigmaf

    @sigmaf.setter
    def sigmaf(self, value):
        if value <= 0:
            raise ValueError('Sigma0 cannot be less or equal to 0')
        self._sigmaf = value
        self._sigma = self.parametrize_learning_law(self.sigma0 , self._sigmaf, 1, self.n_xmod_learning_epochs, self.sigma_learning_type)

    @property
    def sigma_learning_type(self):
        return self._sigma_learning_type

    @sigma_learning_type.setter
    def sigma_learning_type(self, value):
        if value not in self._learning_tpyes:
            raise ValueError("sigma_learning_type must be 'sigmoid', 'invtime' or 'exp'")
        self._sigma_learning_type = value
        self._sigma = self.parametrize_learning_law(self.sigma0 , self.sigmaf, 1, self.n_xmod_learning_epochs, self._sigma_learning_type)

    @property
    def alpha0(self):
        return self._alpha0

    @alpha0.setter
    def alpha0(self, value):
        if value <= 0:
            raise ValueError('Alpha0 cannot be less or equal to 0')
        self._alpha0 = value
        self._alpha = self.parametrize_learning_law(self._alpha0 , self.alphaf, 1, self.n_xmod_learning_epochs, self.alpha_learning_type)

    @property
    def alphaf(self):
        return self._alphaf

    @alphaf.setter
    def alphaf(self, value):
        if value <= 0:
            raise ValueError('Alphaf cannot be less or equal to 0')
        self._alphaf = value
        self._alpha = self.parametrize_learning_law(self.alpha0 , self._alphaf, 1, self.n_xmod_learning_epochs, self.alpha_learning_type)

    @property
    def alpha_learning_type(self):
        return self._alpha_learning_type

    @alpha_learning_type.setter
    def alpha_learning_type(self, value):
        if value not in self._learning_tpyes:
            raise ValueError("alpha_learning_type must be 'sigmoid', 'invtime' or 'exp'")
        self._alpha_learning_type = value
        self._alpha = self.parametrize_learning_law(self.alpha0 , self.alphaf, 1, self.n_xmod_learning_epochs, self._alpha_learning_type)

    def parametrize_learning_law(self, v0, vf, t0, tf, learning_type):
        assert learning_type in ('sigmoid', 'invtime', 'exp'), "learning_type must be 'sigmoid', 'invtime' or 'exp'"

        #Matrix zur Speicherung der parametrisierten Lernrate für jeden Schritt
        y = np.zeros((tf - t0,))

        #Zeitschritte von 1 bis tf
        t = np.array([i for i in range(1, tf +1 )])

        #Unterscheidung nach Lernarten (sigmoid, invtime, exp)
        if learning_type == 'sigmoid':
            s = -np.floor(np.log10(tf)) * 10**(-(np.floor(np.log10(tf))))
            p = abs(s*10**(np.floor(np.log10(tf)) + np.floor(np.log10(tf))/2))
            y = v0 - (v0)/(1+np.exp(s*(t-(tf/p)))) + vf

        elif learning_type == 'invtime':
            B = (vf * tf - v0 * t0) / (v0 - vf)
            A = v0 * t0 + B * v0
            y = [A / (t[i] + B) for i in range(len(t))]

        elif learning_type == 'exp':
            if v0 < 1:
                p = -np.log(v0)
            else:
                p = np.log(v0)
            y = v0 * np.exp(-t/(tf/p))

        return y

    def get_winput_ensemble_array(self):
        return nengo.networks.EnsembleArray(
            n_neurons=self.n_neurons*self.n_lif_per_neuron, 
            n_ensembles=self.n_neurons, 
            radius=self.radius, 
            neuron_type=nengo.LIF(tau_rc=self.tau_rc, tau_ref=self.tau_ref)
        )

    def get_activity_ensemble_array(self):
        return nengo.networks.EnsembleArray(
            n_neurons=self.n_neurons*self.n_lif_per_neuron, 
            n_ensembles=self.n_neurons, 
            radius=self.radius, 
            neuron_type=nengo.LIF(tau_rc=self.tau_rc, tau_ref=self.tau_ref)
        )

    def get_std_ensemble_array(self):
        return nengo.networks.EnsembleArray(
            n_neurons=self.n_neurons*self.n_lif_per_neuron, 
            n_ensembles=self.n_neurons, 
            radius=self.radius, 
            intercepts=nengo.dists.Uniform(0.,1.), 
            encoders=nengo.dists.Choice([[1]]), 
            max_rates=nengo.dists.Uniform(400,500),
            neuron_type=nengo.LIF(tau_rc=self.tau_rc, tau_ref=self.tau_ref)
        )

    def build_network(self, sensory_data, learning_rule):
        assert sensory_data.shape[0] == 2

        self._xmod_weights = []
        wcross = uniform(0, 1, (self.n_neurons, self.n_neurons))
        for _ in range(self._n_som):
            self._xmod_weights.append(wcross / wcross.sum())
        self._simulator = None

        if learning_rule == 'covariance':
            learning_process = processes.CovarianceLearning(dt=self.dt, sensory_data=sensory_data, xmod_weights=self._xmod_weights, alpha=self._alpha, sigma=self._sigma, 
                                                            n_neurons=self.n_neurons, n_inner_learning_epochs=self.n_inner_learning_epochs, eta=self.eta, xi=self.xi, sigma_def=self.sigma_def)
        elif learning_rule == 'hebbian':
            learning_process = processes.HebbianLearning(dt=self.dt, sensory_data=sensory_data, xmod_weights=self._xmod_weights, alpha=self._alpha, sigma=self._sigma, 
                                                         n_neurons=self.n_neurons, n_inner_learning_epochs=self.n_inner_learning_epochs, eta=self.eta, xi=self.xi, sigma_def=self.sigma_def)
        elif learning_rule == 'oja':
            learning_process = processes.OjaLearning(dt=self.dt, sensory_data=sensory_data, xmod_weights=self._xmod_weights, alpha=self._alpha, sigma=self._sigma, 
                                                     n_neurons=self.n_neurons, n_inner_learning_epochs=self.n_inner_learning_epochs, eta=self.eta, xi=self.xi, sigma_def=self.sigma_def)
        else:
            raise ValueError("learning_rule must be 'covariance', 'hebbian' or 'oja'")
        
        self._model = nengo.Network()
        with self._model:
            #Repräsentation der WInput-Vektoren als Ensembles
            self._winput_x = self.get_winput_ensemble_array()
            self._winput_y = self.get_winput_ensemble_array()

            #Repräsentation der Aktivitäts-Vektoren als Ensembles
            self._activity_x = self.get_activity_ensemble_array()
            self._activity_y = self.get_activity_ensemble_array()

            #Repräsentation der std-Vektoren als Ensembles
            self._std_x = self.get_std_ensemble_array()
            self._std_y = self.get_std_ensemble_array()

            self._learn_node = nengo.Node(learning_process, size_in=self.n_neurons*self._n_som*3, size_out=self.n_neurons*self._n_som*3)

            #Verbindungen von Neuronen der Ensembles zur Lernmethode ...
            nengo.Connection(self._winput_x.output, self._learn_node[:self.n_neurons])
            nengo.Connection(self._winput_y.output, self._learn_node[self.n_neurons:self.n_neurons*2])
            nengo.Connection(self._activity_x.output, self._learn_node[self.n_neurons*2:self.n_neurons*3])
            nengo.Connection(self._activity_y.output, self._learn_node[self.n_neurons*3:self.n_neurons*4])
            nengo.Connection(self._std_x.output, self._learn_node[self.n_neurons*4:self.n_neurons*5])
            nengo.Connection(self._std_y.output, self._learn_node[self.n_neurons*5:self.n_neurons*6])
            # ... und wieder zurück
            nengo.Connection(self._learn_node[:self.n_neurons], self._winput_x.input)
            nengo.Connection(self._learn_node[self.n_neurons:self.n_neurons*2], self._winput_y.input)
            nengo.Connection(self._learn_node[self.n_neurons*2:self.n_neurons*3], self._activity_x.input)
            nengo.Connection(self._learn_node[self.n_neurons*3:self.n_neurons*4], self._activity_y.input)
            nengo.Connection(self._learn_node[self.n_neurons*4:self.n_neurons*5], self._std_x.input)
            nengo.Connection(self._learn_node[self.n_neurons*5:self.n_neurons*6], self._std_y.input)
            
            #Proben
            self._p_winput_x = nengo.Probe(self._winput_x.output)
            self._p_winput_y = nengo.Probe(self._winput_y.output)
            self._p_activity_x = nengo.Probe(self._activity_x.output)
            self._p_activity_y = nengo.Probe(self._activity_y.output)
            self._p_std_x = nengo.Probe(self._std_x.output)
            self._p_std_y = nengo.Probe(self._std_y.output)

        return self
    
    def execute_simulation(self):
        if self._model is None:
            raise AssertionError("'build_network' has to be called before excuting the simulation")
        self._simulator = nengo.Simulator(self._model, dt=self.dt)
        with self._simulator:
            self._simulator.run_steps(self.n_xmod_learning_epochs)
        return self
    
    def get_probes(self):
        if self._simulator is None:
            raise AssertionError("'execute_simulation' has to be called before returning the results")
        probes = {}
        probes['winput_x'] = self._simulator.data[self._p_winput_x]
        probes['winput_y'] = self._simulator.data[self._p_winput_y]
        probes['activity_x'] = self._simulator.data[self._p_activity_x]
        probes['activity_y'] = self._simulator.data[self._p_activity_y]
        probes['std_x'] = self._simulator.data[self._p_std_x]
        probes['std_y'] = self._simulator.data[self._p_std_y]
        probes['xmod_weights'] = self._xmod_weights
        return probes

