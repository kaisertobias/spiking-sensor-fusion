import nengo
import numpy as np


class LearnProcessBase(nengo.Process):
    def __init__(self, dt, sensory_data, xmod_weights, alpha, sigma, n_neurons=100, n_inner_learning_epochs=50, eta=1., xi=1e-3, sigma_def=.045):
        super().__init__(default_size_in=0, default_size_out=1, default_dt=.001, seed=None)
        self._dt = dt
        self._sensory_data = sensory_data
        self._xmod_weights = xmod_weights
        self._alpha = alpha
        self._sigma = sigma
        self._n_som = 2
        self._n_neurons = n_neurons
        self._n_inner_learning_epochs = n_inner_learning_epochs
        self._eta = eta
        self._xi = xi
        self._sigma_def = sigma_def

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if value <= 0:
            raise ValueError('Simulation step width less or equal than 0 is not possible')
        self._dt = value

    @property
    def sensory_data(self):
        return self._sensory_data

    @sensory_data.setter
    def sensory_data(self, value):
        if value is None:
            raise ValueError('No sensory data given')
        self._sensory_data = value

    @property
    def xmod_weights(self):
        return self._xmod_weights

    @xmod_weights.setter
    def xmod_weights(self, value):
        if value is None:
            raise ValueError('No xmod-weights-matrix given')
        self._xmod_weights = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value is None:
            raise ValueError('No array for alpha given')
        self._alpha = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value is None:
            raise ValueError('No array for sigma given')
        self._sigma = value

    @property
    def n_neurons(self):
        return self._n_neurons

    @n_neurons.setter
    def n_neurons(self, value):
        if value <= 0:
            raise ValueError('Number of neurons less or equal than 0 is not possible')
        self._n_neurons = value

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
    def eta(self, value):
        self._xi = value

    @property
    def sigma_def(self):
        return self._sigma_def

    @sigma_def.setter
    def sigma_def(self, value):
        self._sigma_def = value

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, x):
            epoch = int(t/self.dt)

            if epoch == 0: #wenn erster Durchlauf, Werte der Neuronen initialisieren
                winputs = np.zeros((self._n_som, self.n_neurons))
                activities = np.zeros((self._n_som, self.n_neurons))
                stds = np.ones((self._n_som, self.n_neurons)) * self.sigma_def
            else:
                winputs = np.array([x[:self.n_neurons], x[self.n_neurons:self.n_neurons*2]])
                activities = np.array([x[self.n_neurons*2:self.n_neurons*3], x[self.n_neurons*3:self.n_neurons*4]])
                stds = np.array([x[self.n_neurons*4:self.n_neurons*5], x[self.n_neurons*5:self.n_neurons*6]])
            
            #Inneres Lernen
            if epoch <= self.n_inner_learning_epochs:
                (winputs, activities, stds) = self.inner_learning(winputs, activities, stds, epoch)

            #XMOD-Lernen
            activities = self.xmod_learning(winputs, activities, stds, epoch)

            return np.concatenate((winputs.flatten(), activities.flatten(), stds.flatten()), axis=None)
        return step
    
    #für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def update_activity_vector(self, winput, activity, std, datapoint):
        #Initialisierung des Aktivitätsvektors auf 0
        act_cur = np.zeros((self.n_neurons,))

        #hier kann teilen durch 0 auftreten, weshalb 0-Werte ersetzt werden
        std = np.maximum(std, 1e-8)
        act_cur = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-np.square(datapoint - winput) / (2 * np.square(std)))

        #Normalisierung des Aktivitätsvektors der Population
        if act_cur.sum() != 0:
            act_cur /= act_cur.sum()

        #Aktualisierung der Aktivität für die nächste Iteration
        activity = (1 - self.eta) * activity + self.eta * act_cur
        return activity
    
    #für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def inner_learning(self, winputs, activities, stds, epoch):
        #Kernelwert (Differenz zw. Pos. des aktuellen Neurons und der Pos. des Neurons mit der höchsten Aktivität)
        hwi = np.zeros((self.n_neurons,))

        #Werte der Lernraten bleiben bei Schleifendurchläufe konstant und müssen so nur einmal gelesen werden
        alpha = self.alpha[epoch-1]
        sigma = self.sigma[epoch-1]

        #Faktor ändert sich nicht in Schleifendurchläufen und muss nur einmal berechnet werden
        factor = 1 / (np.sqrt(2 * np.pi) * sigma)

        for _, datapoint in enumerate(self.sensory_data): #iterieren über Datenpunkte
            for i in range(self._n_som):
                # update the activity for the next iteration
                activities[i] = self.update_activity_vector(winputs[i], activities[i], stds[i], datapoint[i])

                #Bestimmung des Neurons mit höchster Aktivität (Gewinnerneuron)
                win_pos = np.argmax(activities[i])

                #Berechnung des Kernelwerts für Neuronen
                #einfacher Gauß'scher Kernel ohne Berücksichtigung der Begrenzungen
                hwi = np.exp(-np.square(np.arange(self.n_neurons) - win_pos) / (2 * sigma**2))

                #Aktualisierung der Gewichtungen der Neuronen
                winputs[i] += alpha * hwi * (datapoint[i] - winputs[i])

                #Aktualisierungen der std der Neuronen
                stds[i] += alpha * factor * hwi * ((datapoint[i] - winputs[i])**2 - stds[i]**2)
        return (winputs, activities, stds)
    
    def xmod_learning(self, winputs, activities, stds, epoch):
        raise NotImplementedError
    

class CovarianceLearning(LearnProcessBase):
    def __init__(self, dt, sensory_data, xmod_weights, alpha, sigma, n_neurons=100, n_inner_learning_epochs=50, eta=1, xi=1e-3, sigma_def=0.045):
        super().__init__(dt, sensory_data, xmod_weights, alpha, sigma, n_neurons, n_inner_learning_epochs, eta, sigma_def)

    #Covariance-Learning für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def xmod_learning(self, winputs, activities, stds, epoch):
        #  mean activities for covariance learning
        avg_act = np.zeros((self.n_neurons, self._n_som))

        # Berechnung des Abfalls für den Mittelwert
        omega = .002 + .998 / (epoch + 2)

        for _, datapoint in enumerate(self.sensory_data):
            for i in range(self._n_som):
                #Aktualisierung des Aktivitätsvektor
                activities[i] = self.update_activity_vector(winputs[i], activities[i], stds[i], datapoint[i])
                # Berechnung des Abfalls für den Mittelwert
                avg_act[:, i] = (1 - omega) * avg_act[:, i] + omega * activities[i][:]

            #Kreuzmodale Hebb'sche Kovarianz-Lernregel: Aktualisierung der Gewichte basierenf auf Kovarianz
            self.xmod_weights[0] = (1 - self._xi) * self.xmod_weights[0] + self._xi * (activities[0] - \
                                   avg_act[:, 0].reshape(self.n_neurons,1)) * (activities[1] - avg_act[:,1].reshape(self.n_neurons,1)).T
            self.xmod_weights[1] = (1 - self._xi) * self.xmod_weights[1] + self._xi * (activities[1] - \
                                   avg_act[:, 1].reshape(self.n_neurons,1)) * (activities[0] - avg_act[:, 0].reshape(self.n_neurons,1)).T
        return activities


class HebbianLearning(LearnProcessBase):
    def __init__(self, dt, sensory_data, xmod_weights, alpha, sigma, n_neurons=100, n_inner_learning_epochs=50, eta=1, xi=1e-3, sigma_def=0.045):
        super().__init__(dt, sensory_data, xmod_weights, alpha, sigma, n_neurons, n_inner_learning_epochs, eta, xi, sigma_def)

    #Hebbian-Learning für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def xmod_learning(self, winputs, activities, stds, epoch):
        for _, datapoint in enumerate(self.sensory_data):
            #Aktualisierung des Aktivitätsvektor
            for i in range(self._n_som):
                activities[i] = self.update_activity_vector(winputs[i], activities[i], stds[i], datapoint[i])

            #Hebb'sche Regel für Kreuzmodalität: Multiplikation der Aktivitäten
            self.xmod_weights[0] = (1 - self.xi) * self.xmod_weights[0] + self.xi * activities[0] * activities[1].T
            self.xmod_weights[1] = (1 - self.xi) * self.xmod_weights[1] + self.xi * activities[1] * activities[0].T
        return activities
    

class OjaLearning(LearnProcessBase):
    def __init__(self, dt, sensory_data, xmod_weights, alpha, sigma, n_neurons=100, n_inner_learning_epochs=50, eta=1, xi=1e-3, sigma_def=0.045):
        super().__init__(dt, sensory_data, xmod_weights, alpha, sigma, n_neurons, n_inner_learning_epochs, eta, xi, sigma_def)

    #Oja-Learning für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def xmod_learning(self, winputs, activities, stds, epoch):
        for _, datapoint in enumerate(self.sensory_data):
            #Aktualisierung des Aktivitätsvektor
            for i in range(self._n_som):
                activities[i] = self.update_activity_vector(winputs[i], activities[i], stds[i], datapoint[i])

            # Oja'sche lokale PCA-Lernregel
            self.xmod_weights[0] = ((1 - self.xi) * self.xmod_weights[0] + self.xi * activities[0] * activities[1].T) / \
                                   np.sqrt(sum(sum((1 - self.xi) * self.xmod_weights[0] + self.xi * activities[0] * activities[1].T)))
            self.xmod_weights[1] = ((1 - self.xi) * self.xmod_weights[1] + self.xi * activities[1] * activities[0].T) / \
                                   np.sqrt(sum(sum((1 - self.xi) * self.xmod_weights[1] + self.xi * activities[1] * activities[0].T)))
        return activities


