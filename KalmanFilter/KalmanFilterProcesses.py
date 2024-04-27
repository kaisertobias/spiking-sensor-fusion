import nengo
import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky


class BaseProcess(nengo.Process):
    """
    The `BaseProcess` class serves as an abstract base class for defining different processes in a simulation
    environment using the Nengo framework. It encapsulates managing the dimensionality of the process and 
    transforming between different representational spaces

    Attributes
    ----------
    _dim_x : int
        The dimensionality of the process state.
    _P_size : int
        The size of the covariance matrix flattened into a unique array considering its symmetry.

    Methods
    -------
    to_model_space(x, P)
        Converts the state `x` and covariance matrix `P` into a flattened form suitable for
        simulation in the nengo Framework.
    from_model_space(data)
        Abstract method to reconstruct state and covariance from the nengo representation.
    make_step(shape_in, shape_out, dt, rng, state=None)
        Abstract method to define the step function of the process for simulation.
    """

    def __init__(self, dim_x):
        """
        Parameters
        ----------
        dim_x : int
            The dimensionality of the process state.
        """
        super(BaseProcess, self).__init__()
        self._dim_x = dim_x
        self._P_size = int(dim_x *(dim_x + 1) / 2)

    def to_model_space(self, x, P):
        return np.concatenate((x.flatten(), P[np.triu_indices(self._dim_x)].flatten()), axis=None)
    
    def from_model_space(self, data):
        x = np.asarray(data[:self._dim_x]).reshape((self._dim_x,))
        P_indices = np.triu_indices(self._dim_x)
        P = np.zeros((self._dim_x, self._dim_x))
        P[P_indices] = data[self._dim_x:self._dim_x+self._P_size]
        P += np.triu(P, 1).T
        return (x, P)
    
    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        raise NotImplementedError
    


class PredictProcess(BaseProcess):
    """
    The `PredictProcess` class extends `BaseProcess`, focusing on the prediction phase of the Kalman-Filter algorithms.

    Attributes
    ----------
    _dim_u : int
        The dimensionality of the control input.

    Methods
    -------
    from_model_space(data)
        Reconstructs the state `x`, covariance matrix `P`, and control input `u` from a flattened representation in 
        the model space. The method overrides the abstract method from `BaseProcess`.
    make_step(shape_in, shape_out, dt, rng, state=None)
        Abstract method to implement the specific prediction logic of Kalman-Filter.
    """


    def __init__(self, dim_x, dim_u):
        """
        Parameters
        ----------
        dim_x : int
            The dimensionality of the process state.
        dim_u : int
            The dimensionality of the control input.
        """
        super(PredictProcess, self).__init__(dim_x)
        self._dim_u = dim_u

    def from_model_space(self, data):
        x, P = super().from_model_space(data)
        u = np.asarray(data[self._dim_x+self._P_size:]).reshape((self._dim_u,))
        return (x, P, u)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        raise NotImplementedError
    


class UpdateProcess(BaseProcess):
    """
    The `UpdateProcess` class extends `BaseProcess`, focusing on the update phase of the Kalman-Filter algorithms.

    Attributes
    ----------
    _dim_z : int
        The dimensionality of the measurement input.

    Methods
    -------
    from_model_space(data)
        Reconstructs the state `x`, covariance matrix `P`, and measurement `z` from a flattened representation in 
        the model space. Overrides the abstract method from `BaseProcess`.
    make_step(shape_in, shape_out, dt, rng, state=None)
        Abstract method to implement the specific update logic of Kalman-Filter.
    """


    def __init__(self, dim_x, dim_z):
        """
        Parameters
        ----------
        dim_x : int
            The dimensionality of the process state.
        dim_z : int
            The dimensionality of the measurement input.
        """
        super(UpdateProcess, self).__init__(dim_x)
        self._dim_z = dim_z

    def from_model_space(self, data):
        x, P = super().from_model_space(data)
        z = np.asarray(data[self._dim_x+self._P_size:]).reshape((self._dim_z,))
        return (x, P, z)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        raise NotImplementedError



class LinearKF_Predict(PredictProcess):
    """
    The `LinearKF_Predict` class implements the prediction step of a linear Kalman-Filter. It extends`PredictProcess`
    and should be used for linear systems. This class encapsulates the linear prediction model and
    allows for the prediction of the state and covariance matrix based on the process dynamics and noise.

    Attributes
    ----------
    _F : numpy.ndarray
        The state transition matrix.
    _Q : numpy.ndarray
        The process noise covariance matrix.
    _B : numpy.ndarray or None
        The control input matrix, optional.

    Methods
    -------
    predict_state(x, u=None)
        Computes the predicted state using the linear state transition model.
    predict_cov_matrix(P)
        Computes the predicted covariance matrix based on the state transition and process noise.
    make_step(shape_in, shape_out, dt, rng, state=None)
        Implements the prediction step logic, using the model space data to predict the next state 
        and covariance, and returns them in model space format.
    """


    def __init__(self, dim_x, F, Q, B=None, dim_u=0):
        """
        Parameters
        ----------
        dim_x : int
            The dimensionality of the process state.
        F : numpy.ndarray
            The state transition matrix.
        Q : numpy.ndarray
            The process noise covariance matrix.
        B : numpy.ndarray, optional
            The control input matrix (default is None).
        dim_u : int, optional
            The dimensionality of the control input (default is 0).
        """
        super(LinearKF_Predict, self).__init__(dim_x, dim_u)
        self._F = F
        self._Q = Q
        self._B = B

    def predict_state(self, x, u=None):
        if self._B != None and u != None:
            return self._F @ x + self._B @ u
        else:
            return self._F @ x
    
    def predict_cov_matrix(self, P):
        return self._F @ P @ self._F.T + self._Q

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, data):
            x, P, u = self.from_model_space(data)
            
            x = self.predict_state(x, u)
            P = self.predict_cov_matrix(P)

            return self.to_model_space(x, P)
        return step



class LinearKF_Update(UpdateProcess):
    """
    The `LinearKF_Update` class implements the update step of a linear Kalman-Filter. It extends`PredictProcess`
    and should be used for linear systems. This class encapsulates the linear update model and
    allows for the update of the state and covariance matrix based on the process dynamics and noise.

    Attributes
    ----------
    _H : numpy.ndarray
        The measurement matrix.
    _R : numpy.ndarray
        The measurement noise covariance matrix.

    Methods
    -------
    compute_kalman_gain(P)
        Calculates the Kalman gain matrix based on the covariance estimate `P` and the measurement model.
    compute_residual(x, z)
        Computes the difference between the actual measurement `z` and the predicted measurement.
    update_state(x, K, y)
        Updates the state estimate `x` using the Kalman gain `K` and the measurement residual `y`.
    update_cov_matrix(P, K)
        Updates the covariance matrix `P` based on the Kalman gain `K`.
    make_step(shape_in, shape_out, dt, rng, state=None)
        Implements the update step logic, using the model space data to update the state 
        and covariance based on the measurement, and returns them in model space format.
    """

    def __init__(self, dim_x, dim_z, H, R):
        """
        Parameters
        ----------
        dim_x : int
            The dimensionality of the process state.
        dim_z : int
            The dimensionality of the measurement input.
        H : numpy.ndarray
            The measurement matrix.
        R : numpy.ndarray
            The measurement noise covariance matrix.
        """
        super(LinearKF_Update, self).__init__(dim_x, dim_z)
        self._H = H
        self._R = R

    def compute_kalman_gain(self, P):
        S = self._H @ P @ self._H.T + self._R
        K = P @ self._H.T @ inv(S)
        return K

    def compute_residual(self, x, z):
        return z - self._H @ x
    
    def update_state(self, x, K, y):
        return x + K @ y
    
    def update_cov_matrix(self, P, K):
        I_KH = np.eye(self._dim_x) - K @ self._H
        return I_KH @ P @ I_KH.T + K @ self._R @ K.T

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, data):
            x, P, z = self.from_model_space(data)
            
            K = self.compute_kalman_gain(P)
            y = self.compute_residual(x, z)
            x = self.update_state(x, K, y)
            P = self.update_cov_matrix(P, K)

            return self.to_model_space(x, P)
        return step
    


class EKF_Predict(PredictProcess):
    """
    The `EKF_Predict` class implements the prediction step of an Extended Kalman Filter (EKF) for systems where the
    process model is nonlinear. It extends `PredictProcess` and utilizes a nonlinear function and its Jacobian
    to predict the state and covariance matrix.

    Attributes
    ----------
    _f : callable
        The nonlinear state transition function.
    _Q : numpy.ndarray
        The process noise covariance matrix.

    Methods
    -------
    predict_state(x, u=None)
        Uses the nonlinear state transition function `_f` to predict the next state.
    predict_cov_matrix(P)
        Computes the predicted covariance matrix using the state transition function to approximate 
        the effect of the nonlinearity on the covariance.
    make_step(shape_in, shape_out, dt, rng, state=None)
        Implements the prediction step logic for a nonlinear system, using the provided state and covariance to predict
        their next values and returning them in model space format.
    """

    def __init__(self, dim_x, f, F_Jacobian, Q, dim_u=0):
        """
        Parameters
        ----------
        dim_x : int
            The dimensionality of the process state.
        f : callable
            The nonlinear state transition function.
        F_Jacobian : callable
            Nonlinear function that computes matrix F for current state estimate.
        Q : numpy.ndarray
            The process noise covariance matrix.
        dim_u : int, optional
            The dimensionality of the control input (default is 0).
        """
        super(EKF_Predict, self).__init__(dim_x, dim_u)
        self._f = f
        self._F_Jacobian = F_Jacobian
        self._Q = Q

    def predict_state(self, x, u):
        return self._f(x, u)
    
    def predict_cov_matrix(self, P, F):
        return F @ P @ F.T + self._Q

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, data):
            x, P, u = self.from_model_space(data)
            
            F = self._F_Jacobian(x)
            x = self.predict_state(x, u)
            P = self.predict_cov_matrix(P, F)

            return self.to_model_space(x, P)
        return step
    


class EKF_Update(UpdateProcess):
    """
    The `EKF_Update` class implements the update step of an Extended Kalman Filter (EKF) for systems where the
    measurement model is nonlinear. It extends `UpdateProcess` and uses a nonlinear measurement function along with
    its Jacobian to update the state and covariance estimates based on new measurements.

    Attributes
    ----------
    _h : callable
        The nonlinear measurement function.
    _H_Jacobian : callable
        A function returning the Jacobian of the measurement function `_h`.
    _R : numpy.ndarray
        The measurement noise covariance matrix.

    Methods
    -------
    compute_kalman_gain(P, H)
        Calculates the Kalman gain matrix using the covariance matrix `P` and the Jacobian of the measurement 
        function `H`, considering the measurement noise.
    compute_residual(x, z)
        Computes the residual between the actual measurement `z` and the measurement predicted by the nonlinear 
        measurement function `_h`.
    update_state(x, K, y)
        Updates the state estimate using the Kalman gain `K` and the measurement residual `y`.
    update_cov_matrix(P, K, H)
        Updates the covariance matrix after incorporating the measurement.
    make_step(shape_in, shape_out, dt, rng, state=None)
        Implements the update step logic for a nonlinear system, adjusting the state and covariance based on the 
        new measurement and returning the updated values in model space format.
    """


    def __init__(self, dim_x, dim_z, h, H_Jacobian, R):
        """
        Parameters
        ----------
        dim_x : int
            The dimensionality of the process state.
        dim_z : int
            The dimensionality of the measurement input.
        h : callable
            The nonlinear measurement function.
        H_Jacobian : callable
            A function returning the Jacobian of the measurement function `h`.
        R : numpy.ndarray
            The measurement noise covariance matrix.
        """
        super(EKF_Update, self).__init__(dim_x, dim_z)
        self._h = h
        self._H_Jacobian = H_Jacobian
        self._R = R

    def compute_kalman_gain(self, P, H):
        S = H @ P @ H.T + self._R
        K = P @ H.T @ inv(S)
        return K

    def compute_residual(self, x, z):
        return z - self._h(x)
    
    def update_state(self, x, K, y):
        return x + K @ y
    
    def update_cov_matrix(self, P, K, H):
        I_KH = np.eye(self._dim_x) - (K @ H)
        return (I_KH @ P @ I_KH.T) + (K @ self._R @ K.T)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, data):
            x, P, z = self.from_model_space(data)
            
            H = self._H_Jacobian(x)
            K = self.compute_kalman_gain(P, H)
            y = self.compute_residual(x, z)
            x = self.update_state(x, K, y)
            P = self.update_cov_matrix(P, K, H)

            return self.to_model_space(x, P)
        return step



class UKF_Base(object):
    """
    The `UKF_Base` class provides common functionalities needed by both the prediction and update phases of the UKF,
    focusing on the handling of sigma points and the computation of mean and covariance from these points.

    Attributes
    ----------
    _W : numpy.ndarray
        Weights associated with sigma points for computing means and covariances.

    Methods:
    compute_mean_and_covariance(sigmas, noise)
        Computes the weighted mean and covariance of the transformed sigma points `sigmas`, incorporating additional 
        process or measurement noise `noise`. This method allows the estimation of state and covariance from
        the nonlinear transformations of sigma points.
    """

    def __init__(self, W):
        """"
        Parameters
        ----------
        W : numpy.ndarray
            Weights associated with sigma points for computing means and covariances.
        """
        super(UKF_Base, self).__init__()
        self._W = W
    
    def compute_mean_and_covariance(self, sigmas, noise):
        #Berechnung des Mittelwerts der Sigma-Punkte
        mean = self._W @ sigmas
        #Berechnung des residuals
        y = sigmas - mean[np.newaxis, :]
        cov = y.T @ np.diag(self._W) @ y + noise
        return (mean, cov)



class UKF_Predict(PredictProcess, UKF_Base):
    """
    The `UKF_Predict` class implements the prediction step of the Unscented Kalman Filter (UKF) by extending
    `PredictProcess` and `UKF_Base`. It is designed to handle nonlinear process models by using sigma points to
    approximate the state distribution.

    Attributes
    ----------
    _sigmas : numpy.ndarray
        Sigma points array for the state.
    _sigmas_f : numpy.ndarray
        Transformed sigma points through the process model.
    _f : callable
        The nonlinear state transition function.
    _Q : numpy.ndarray
        The process noise covariance matrix.
    _dt : float
        The time step width.

    Methods
    -------
    compute_cholesky_decomposition(P)
        Computes the Cholesky decomposition of the covariance matrix `P` for sigma point generation.
    compute_sigmas(x, P)
        Generates sigma points around the current state `x` and covariance `P`.
    compute_f_of_sigmas()
        Applies the state transition function `_f` to each sigma point to predict their next positions.
    make_step(shape_in, shape_out, dt, rng, state=None)
        Implements the prediction logic for the UKF, transforming sigma points through the process model, 
        computing the predicted state and covariance, and returning them in model space format.
    """

    def __init__(self, dim_x, W, kappa, sigmas, sigmas_f, f, Q, dim_u=0):
        """
        Parameters
        ----------
        dim_x : int
            The dimensionality of the process state.
        W : numpy.ndarray
            Weights associated with sigma points for computing means and covariances.
        sigmas : numpy.ndarray
            Sigma points array for the state.
        sigmas_f : numpy.ndarray
            Transformed sigma points through the process model.
        f : callable
            The nonlinear state transition function.
        Q : numpy.ndarray
            The process noise covariance matrix.
        dt : float
            The time step width.
        dim_u : int, optional
            The dimensionality of the control input (default is 0).
        """
        PredictProcess.__init__(self, dim_x=dim_x, dim_u=dim_u)
        UKF_Base.__init__(self, W=W)
        self._kappa = kappa
        self._sigmas = sigmas
        self._sigmas_f = sigmas_f
        self._f = f
        self._Q = Q

    def eigenvalue_decomposition(self, P):
        eigenvalues, U = np.linalg.eigh((self._dim_x + self._kappa) * P)
        lambda_prime = np.diag(np.maximum(eigenvalues, 0))
        lambda_prime_sqrt = np.sqrt(lambda_prime)
        _, R = np.linalg.qr((U @ lambda_prime_sqrt).T)
        return R

    def cholesky_decomposition(self, P):
        try:
            U = cholesky((self._dim_x + self._kappa) * P)
        except np.linalg.LinAlgError:
            U = self.eigenvalue_decomposition(P)
        return U
    
    def compute_sigmas(self, x, P):
        U = self.cholesky_decomposition(P)
        sigmas = np.zeros((2*self._dim_x+1, self._dim_x))
        sigmas[0] = x
        for i in range(self._dim_x):
            sigmas[i+1] = x + U[i]
            sigmas[self._dim_x+i+1] = x - U[i]
        return sigmas
        
    def compute_f_of_sigmas(self, u, t):
        for i, s in enumerate(self._sigmas):
            self._sigmas_f[i] = self._f(s, t)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, data):
            x, P, u = self.from_model_space(data)

            self._sigmas = self.compute_sigmas(x, P)
            self.compute_f_of_sigmas(u, t)
            x, P = self.compute_mean_and_covariance(self._sigmas_f, self._Q)
            self._sigmas_f = self.compute_sigmas(x, P)

            return self.to_model_space(x, P)
        return step



class UKF_Update(UpdateProcess, UKF_Base):
    """
    The `UKF_Update` class implements the update step of the Unscented Kalman Filter (UKF) by extending
    `UpdateProcess` and `UKF_Base`. It is designed to handle nonlinear measurement models by utilizing sigma points
    to approximate the posterior state distribution given new measurements.

    Attributes
    ----------
    _sigmas_f : numpy.ndarray
        Sigma points transformed through the process model.
    _sigmas_h : numpy.ndarray
        Sigma points transformed through the measurement model.
    _h : callable
        The nonlinear measurement function.
    _R : numpy.ndarray
        The measurement noise covariance matrix.

    Methods
    -------
    compute_h_of_sigmas()
        Applies the measurement function `_h` to each sigma point to predict their measurement positions.
    compute_cross_variance(x, zp): 
        Computes the cross-covariance matrix between the state sigma points and the measurement sigma points.
    compute_klaman_gain(Pxz, S)
        Calculates the Kalman gain matrix using the cross-covariance matrix `Pxz` and the measurement covariance matrix `S`.
    update_state(x, K, z, zp)
        Updates the state estimate using the Kalman gain `K` and the measurement residual.
    update_cov_matrix(P, K, S)
        Updates the covariance matrix after incorporating the measurement.
    make_step(shape_in, shape_out, dt, rng, state=None)
        Implements the update logic for the UKF, processing sigma points through the measurement model, updating the state and 
        covariance based on the new measurement, and returning the updated values in model space format.
    """

    def __init__(self, dim_x, dim_z, W, sigmas_f, sigmas_h, h, R):
        """
        Parameters
        ----------
        dim_x : int
            The dimensionality of the process state.
        dim_z : int
            The dimensionality of the measurement input.
        W : numpy.ndarray
            Weights associated with sigma points for computing means and covariances.
        sigmas_f : numpy.ndarray
            Sigma points transformed through the process model.
        sigmas_h : numpy.ndarray
            Sigma points transformed through the measurement model.
        h : callable
            The nonlinear measurement function.
        R : numpy.ndarray
            The measurement noise covariance matrix.
        """
        UpdateProcess.__init__(self, dim_x=dim_x, dim_z=dim_z)
        UKF_Base.__init__(self, W=W)
        self._sigmas_f = sigmas_f
        self._sigmas_h = sigmas_h
        self._h = h
        self._R = R

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
    
    def update_cov_matrix(self, P, K, S):
        return P - K @ S @ K.T

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, data):
            x, P, z = self.from_model_space(data)

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
            P = self.update_cov_matrix(P, K, S)

            return self.to_model_space(x, P)
        return step


