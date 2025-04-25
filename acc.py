import numpy as np

import casadi as ca


class ACC:
    """
    Adaptive Cruise Control (ACC) system.

    Parameters:
    - CtrlAffineSys: Control affine system class.
    """

    def __init__(self, x0: np.ndarray, **kwargs):
        """
        Initialize the ACC system.
        State:
            - x = [p v z]^T \in R^3
                - p: position
                - v: velocity
                - z: distance to the lead vehicle.

        Control input:
            - u = F \in R^1 wheel force.

        Dynamics:
        \dot{x} = f(x) + g(x) u
        where:
            - f(x) = [v; -1/m * F_r(v); v_0 - v]^T
            - g(x) = [0; 1/m; 0]^T
            - F_r: Rolling resistance force. F_r(v) = f_0 + f_1 * v + f_2 * v^2



        Parameters:
        - x0: Initial state of the system.
        """
        # Constants
        m = kwargs.get("m", 1650)
        g = kwargs.get("g", 9.81)
        v_d = kwargs.get("v_d", 24)
        f_0 = kwargs.get("f_0", 0.1)
        f_1 = kwargs.get("f_1", 5)
        f_2 = kwargs.get("f_2", 0.25)

        def F_r(x: np.ndarray) -> float:
            """
            Rolling resistance force.
            F_r(v) = f_0 + f_1 * v + f_2 * v^2
            """
            return f_0 + f_1 * x[1] + f_2 * x[1] ** 2

        def f(x: np.ndarray) -> np.ndarray:
            """
            Drift term of the system.
            f(x) = [v; -1/m * F_r(v); v_0 - v]^T
            """
            return np.array([x[1], -1 / m * F_r(x), v_d - x[1]])

        def g(x: np.ndarray) -> np.ndarray:
            """
            Control input term of the system.
            g(x) = [0; 1/m; 0]^T
            """
            return np.array([[0], [1 / m], [0]])

        self.f = f
        self.g = g

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


class Controller:
    """
    Constraints:
    - Input constraint
        -m * c_d * g <= u <= m * c_a * g
    - Stability objective: v -> v_d (v_d: desired velocity)
        (v - v_d) * (2/m * (u - F_r + lambda * (v - v_d))) <= delta
    - Safety objective: z >= T_h * v (T_h: time headway)
        1/m * (T_h + (v - v_0)/(c_d * g)) * (F_r - u) + (v - v_) > 0

    Objective:
        (u - u_ref)^T * H * (u - u_ref) + p * delta^2
    """

    def __init__(self, **kwargs):
        self.T_h = kwargs.get("T_h", 1.8)
        self.v_d = kwargs.get("v_d", 24)
        self.v_0 = kwargs.get("v_0", 14)
        self.c_d = kwargs.get("c_d", 0.3)
        self.c_a = kwargs.get("c_a", 0.3)
        self.clf_rate = kwargs.get("clf_rate", 5)  # lambda
        self.cbf_rate = kwargs.get("cbf_rate", 5)  # gamma
        self.m = kwargs.get("m", 1650)
        self.g = kwargs.get("g", 9.81)
        self.input_weight = 2 / (self.m**2)  # H
        self.slack_weight = kwargs.get("slack_weight", 2e-2)  # p

    def nlp_solve(self, t, x0, u_ref):
        opti = ca.Opti()

        # Decision variables
        u = opti.variable()
        delta = opti.variable()

        # Parameters
        x = opti.parameter(3)  # Current state
        u_ref_ = opti.parameter()  # Reference input

        # Rolling resistance force
        F_r = (
            self.m * self.g * self.c_d
            + self.m * self.c_a * x[1]
            + self.m * self.c_a * x[1] ** 2
        )

        # Input constraints
        opti.subject_to(-self.m * self.c_d * self.g <= u)
        opti.subject_to(u <= self.m * self.c_a * self.g)

        # Stability constraint (CLF)
        V_x = (x0[1] - self.v_d) ** 2  # Lyapunov function
        clf = (x[1] - self.v_d) * (
            2 / self.m * (u - F_r + self.clf_rate * (x[1] - self.v_d))
        )
        opti.subject_to(clf <= delta)

        # Safety constraint (CBF)
        B_x = x0[2] - self.T_h * x0[1]  # Barrier function
        cbf = (1 / self.m) * (self.T_h + (x[1] - self.v_0) / (self.c_d * self.g)) * (
            F_r - u
        ) + B_x
        opti.subject_to(cbf >= 0)

        # Objective function
        objective = self.input_weight * (u - u_ref_) ** 2 + self.slack_weight * delta**2
        opti.minimize(objective)

        # Set parameters
        opti.set_value(x, x0)
        opti.set_value(u_ref_, u_ref)

        # Solve the optimization problem
        p_opts = {"expand": True}
        s_opts = {"max_iter": 100}
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
            return sol.value(u), V_x, B_x, sol.value(delta)
        except RuntimeError:
            return None, None, None, None
