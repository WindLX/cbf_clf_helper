import numpy as np


class CtrlAffineSys:
    def __init__(
        self,
        xdim: int,
        udim: int,
        f: callable,
        g: callable,
        cbf: callable,
        lf_cbf: callable,
        lg_cbf: callable,
        clf: callable,
        lf_clf: callable,
        lg_clf: callable,
        x0: np.ndarray,
    ):
        """
        Initialize a control affine system.

        Parameters:
        - xdim: Dimension of the state space.
        - udim: Dimension of the control input space.
        - f: Function representing the drift term.
        - g: Function representing the control input term.
        - cbf: Function representing the control barrier function.
        - lf_cbf: Function representing the Lie derivative of the CBF.
        - lg_cbf: Function representing the Lie derivative of the CBF with respect to the control input.
        - clf: Function representing the control Lyapunov function.
        - lf_clf: Function representing the Lie derivative of the CLF.
        - lg_clf: Function representing the Lie derivative of the CLF with respect to the control input.
        - x0: Initial state of the system.
        """
        self.xdim = xdim
        self.udim = udim
        self.f = f
        self.g = g
        self.cbf = cbf
        self.lf_cbf = lf_cbf
        self.lg_cbf = lg_cbf
        self.clf = clf
        self.lf_clf = lf_clf
        self.lg_clf = lg_clf
        self.x0 = x0

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the dynamics of the system.

        Parameters:
        - t: Current time.
        - x: Current state of the system.
        - u: Control input.

        Returns:
        - dx: Time derivative of the state.
        """
        return self.f(x) + self.g(x) @ u
