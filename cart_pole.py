from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import GroverOptimizer, MinimumEigenOptimizer
from qiskit.optimization.converters import QuadraticProgramToQubo, QuadraticProgramConverter
from qiskit import Aer, execute
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from docplex.mp.model import Model
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent
import functools
import gc
import random
import math
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.circuit.library import MCMT
from qiskit.providers.aer.noise.errors import depolarizing_error
import pickle

class CartPoleActionModel:
    """
    This class provides methods to calculate the dynamics and cost for a given state and control input,
    as well as methods for binary encoding of control inputs.
    """

    def __init__(self, N: int, M: int, M_global: int, u_min: float, u_max: float,
                 x_min: np.ndarray, x_max: np.ndarray,
                 Q: np.ndarray = np.eye(4), R: float = 1, dt: float = 0.01,
                 x_ref: np.ndarray = np.array([0, 0, 0, 0]), u_ref: float = 0,
                 g: float = 9.81, mc: float = 1.0, mp: float = 0.1, l: float = 0.5):
        """
        Constructs all the necessary attributes for the CartPoleActionModel object.

        Parameters:
            N (int): Number of bits for state (x) binary encoding.
            M (int): Number of bits for control-input (u) binary encoding.
            u_min (float): Minimum control input.
            u_max (float): Maximum control input.
            Q (numpy.ndarray): State cost matrix. Defaults to 4x4 identity matrix.
            R (float): Control cost. Defaults to 1.
            dt (float): Time step for the dynamics. Defaults to 0.01.
            x_ref (numpy.ndarray): Reference state. Defaults to array [0, 0, 0, 0].
            u_ref (float): Reference control input. Defaults to 0.
            g (float): Gravitational acceleration. Defaults to 9.81.
            mc (float): Mass of the cart. Defaults to 1.0.
            mp (float): Mass of the pole. Defaults to 0.1.
            l (float): Length of the pole. Defaults to 0.5.
        """

        self.nx = 4
        self.nu = 1

        self.u_none = np.zeros((self.nu,))

        self.N = N
        self.M = M
        self.M_global = M_global
        self.u_min = u_min
        self.u_max = u_max

        self.x_ref = np.array(x_ref).reshape(self.nx,)
        self.u_ref = np.array(u_ref).reshape(self.nu,)

        self.x_min = x_min.flatten()
        self.x_max = x_max.flatten()

        self.Q = np.array(Q).reshape(self.nx, self.nx)
        self.R = np.array(R).reshape(self.nu, self.nu)

        self.dt = dt
        self.g = g
        self.mc = mc
        self.mp = mp
        self.l = l

        self.total_mass = mc + mp
        self.pole_mass_length = mp * l

        self.Acont = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, self.mp * self.g/self.mc, 0, 0], [0, self.total_mass * self.g / (self.mc * self.l), 0, 0]])
        self.Bcont = np.array([0, 0, 1/self.mc, 1/(self.mc * self.l)]).reshape(self.nx,1)

        temp = self.dt * np.sqrt(self.g * self.total_mass/(self.l * self.mc))

        self.A = np.array([[1, self.mp * self.l * (-1 + np.cosh(temp)) / self.total_mass, self.dt, -self.dt * self.l * self.mp / self.total_mass + np.sqrt(self.l**3 * self.mc) * self.mp * np.sinh(temp) / np.sqrt(self.g * self.total_mass**3)],
                           [0, np.cosh(temp), 0, np.sqrt(self.l * self.mc) * np.sinh(temp) / np.sqrt(self.g * self.total_mass)],
                           [0, np.sqrt(self.g * self.l) * self.mp * np.sinh(temp) / np.sqrt(self.mc * self.total_mass), 1, self.mp * self.l * (-1 + np.cosh(temp)) / self.total_mass],
                           [0, np.sqrt(self.g * self.total_mass) * np.sinh(temp) / np.sqrt(self.l * self.mc), 0, np.cosh(temp)]])

        self.B = np.array([(-2.0 * self.l * self.mp + self.dt**2 * self.g * self.total_mass + 2.0 * self.l * self.mp * np.cosh(temp)) / (2.0 * self.g * self.total_mass**2),
                            (-1.0 + np.cosh(temp)) / (self.g * self.total_mass),
                            self.dt / self.total_mass + np.sqrt(self.l) * self.mp * np.sinh(temp) / np.sqrt(self.g * self.mc * self.total_mass**3),
                            np.sinh(temp) / np.sqrt(self.g * self.l * self.mc * self.total_mass)]).reshape(self.nx, 1)


        self.ws = 1.0
        self.wc = 1.0

        pass

    def calc_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculates the next state of the cart pole system using the system dynamics.

        Parameters:
            x (numpy.ndarray): Current state of the system.
            u (float): Current control input.

        Returns:
            numpy.ndarray: The next state of the system.
        """
        dx = np.zeros((self.nx,))

        theta_dot = x[3].item()

        force = np.array(u).item()
        sintheta = np.sin(x[1].item())
        costheta = np.cos(x[1].item())
        mu_theta = (self.mc + self.mp * sintheta**2)

        x_ddot = (force + self.mp * costheta * sintheta * self.g - self.mp * self.l * sintheta * theta_dot) / mu_theta
        theta_ddot = (force * costheta / self.l + self.total_mass * self.g * sintheta / self.l - self.mp * costheta * sintheta * theta_dot**2) / mu_theta

        # Symplectic Euler Integration
        vel = x[int(self.nx/2):].flatten()
        acc = np.array([x_ddot, theta_ddot]).flatten()

        dx[:int(self.nx/2)] = np.array(vel * self.dt + acc * self.dt**2).flatten()
        dx[int(self.nx/2):] = np.array(acc * self.dt).flatten()

        xnext = x.flatten() + dx

        return xnext


    def calc_cost(self, x: np.ndarray, u: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
        """
        Calculates the cost based on the current state, control input, and their respective references.

        Parameters:
            x (numpy.ndarray): Current state of the system.
            u (float): Current control input.

        Returns:
            float: The calculated cost.
        """

        dx = (self.x_ref - x.flatten()).reshape(self.nx, 1)
        du = (self.u_ref - np.array(u).reshape(self.nu,))

        cost_x = (dx.T @ Q @ dx).item()
        cost_u = (du.T @ R @ du).item()

        dsin = 0
        dcos = 0

        dsin = (np.sin(self.x_ref[1]) - np.sin(x[1]))**2 * self.ws
        dcos = (np.cos(self.x_ref[1]) - np.cos(x[1]))**2 * self.wc

        return cost_x + cost_u + (dsin + dcos)

class FindMinimum:

    def __init__(self, verbose: bool = False) -> None:

        self.initial_threshold = None

        print(f"Initialised Minimum Search Algorithm")

        self.verbose = verbose

        pass


    def create_oracle(self, values: List[float], threshold: float) -> QuantumCircuit:
        """
        Create an oracle circuit for the Grover operator.
        """
        n = (len(values) - 1).bit_length()
        oracle_qc = QuantumCircuit(n + 1)  # n qubits + 1 ancilla

        # print(f"Number of qubits: {n}")
        # print(f"Threshold: {threshold}")

        binary_values = [format(i, f'0{n}b') for i in range(len(values))]

        # print("Binary representations:")
        # for i, bv in enumerate(binary_values):
        #     print(f"Index {i}: {bv}, Value: {values[i]}")

        for i, bin_val in enumerate(binary_values):
            if values[i] <= threshold:
                # print(f"Marking state {i}: {bin_val}")
                for j, b in enumerate(bin_val):
                    if b == '0':
                        oracle_qc.x(j)

                oracle_qc.mcx(list(range(n)), n)  # Multi-controlled X gate

                for j, b in enumerate(bin_val):
                    if b == '0':
                        oracle_qc.x(j)

        return oracle_qc

    def diffusion_operator(self,n: int) -> QuantumCircuit:
        """Create the diffusion operator for Grover's algorithm."""
        qc = QuantumCircuit(n)

        for qubit in range(n):
            qc.h(qubit)
        for qubit in range(n):
            qc.x(qubit)

        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)
        qc.h(n-1)

        for qubit in range(n):
            qc.x(qubit)
        for qubit in range(n):
            qc.h(qubit)

        return qc

    def findMin(self, values: List[float], num_iterations: int = 10, noise_param: float = 0.01, cost_threshold: float = 0.01) -> Tuple[int, float]:
        """
        Implement Grover Adaptive Search to find the minimum value in the array with depolarizing noise.

        :param values: List of values to search
        :param num_iterations: Number of Grover iterations to perform
        :param noise_param: Depolarizing noise parameter
        :return: Index of the minimum value
        """
        n = (len(values) - 1).bit_length()

        # Create a noise model
        noise_model = NoiseModel()

        error_target_only = depolarizing_error(noise_param, 1)


        for control_qubit in range(n):
            noise_model.add_nonlocal_quantum_error(
                error_target_only,
                'cx',
                [control_qubit, n],
                [n]
            )

        threshold = cost_threshold
        best_index = 0



        for iteration in range(num_iterations):
            # print(f"\nIteration {iteration + 1}")
            qr = QuantumRegister(n + 1)
            cr = ClassicalRegister(n)
            qc = QuantumCircuit(qr, cr)

            qc.h(qr[:n])
            qc.x(qr[n])
            qc.h(qr[n])

            oracle = self.create_oracle(values, threshold)
            diffusion = self.diffusion_operator(n)

            qc = qc.compose(oracle)
            qc = qc.compose(diffusion)

            qc.measure(qr[:n], cr)

            backend = Aer.get_backend('qasm_simulator')
            job = execute(qc, backend, noise_model=noise_model, shots=1000)
            result = job.result()
            counts = result.get_counts(qc)

            # print("Measurement results:")
            # for state, count in counts.items():
                # print(f"State {state}: {count} shots")

            valid_counts = {state: count for state, count in counts.items() if int(state, 2) < len(values)}
            if valid_counts:
                max_count = max(valid_counts, key=valid_counts.get)
                index = int(max_count, 2)
            else:
                print("No valid measurements found. Using current best index.")
                index = int(best_index)

            # print(f"Most frequent valid state: {max_count}, Index: {index}")

            if values[index] < values[best_index]:
                best_index = index
            old_threshold = threshold
            threshold = min(threshold, values[best_index])
            # print(f"Threshold updated: {old_threshold} -> {threshold}")
            # print(f"Current best index: {best_index}, value: {values[best_index]}")

        return int(best_index), values[best_index]


class StateNode:
    """
    Represents a node in a state space tree for control and planning algorithms.

    Attributes:
        state (numpy.ndarray): The current state of the node.
        control_input (Optional[float]): The control input that led to this state. None for root node.
        parent (Optional[StateNode]): The parent node in the state space tree. None for root node.
        cost (float): The cumulative cost to reach this node from the root.
        children (List[StateNode]): Child nodes of the current node.
        total_cost (float): The cumulative cost at each step along the path from the root to this node.
    """

    def __init__(self, state: np.ndarray, control_input: Optional[float] = None,
                 parent: Optional['StateNode'] = None, cost: float = 0):
        """
        Initializes a StateNode object.

        Parameters:
            state (numpy.ndarray): The current state of the node.
            control_input (Optional[float]): The control input that led to this state. Defaults to None.
            parent (Optional[StateNode]): The parent node in the state space tree. Defaults to None.
            cost (float): The cumulative cost to reach this node from the root. Defaults to 0.
        """
        self.state = state
        self.control_input = control_input
        self.parent = parent
        self.cost = cost
        self.children = []

        self.total_cost = cost if parent is None else parent.total_cost + cost

        pass


class GAS_DP_Solver:
    """
    A solver class implementing a hybrid algorithm combining local and global search strategies,
    potentially utilizing quantum computing for optimization.

    Attributes:
        action_models (List[<ActionModelType>]): A list of action models used in the solver.
        T (int): The total number of nodes in the solver's graph.
    """

    def __init__(self, action_model_list, N_global: int, M_global: int, verbose: bool = False, debug: bool = False):
        """
        Initializes the GAS_DP_Solver with a list of action models.

        Parameters:
            action_model_list (List[<ActionModelType>]): A list of action models.
        """

        self.action_models = action_model_list

        self.T = len(self.action_models)

        self.debug = debug
        self.verbose = verbose
        # self.exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        if self.debug:
            self.exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        else:
            self.exact_solver = None

        self.qmin_algo = FindMinimum(self.verbose)

        self.N_global = N_global
        self.M_global = M_global

        print(f"Initialized GAS_DP_Solver with {len(self.action_models)} action models.")

        pass

    def calc_cost_offset(self, curAM, nextAM, x: np.ndarray) -> float:

        offset = 0.0
        xk = x.reshape(curAM.nx,1)

        xk_ref = curAM.x_ref.reshape(curAM.nx,1)
        uk_ref = curAM.u_ref.reshape(curAM.nu,1)

        xnext_ref = nextAM.x_ref.reshape(nextAM.nx,1)
        dx_k = (xk_ref - xk).copy()

        # Current Timestep's Offset
        offset += (dx_k.T @ curAM.Q @ dx_k + uk_ref.T @ curAM.R @ uk_ref).item()

        # Next Timestep's Offset Contribution
        offset += (xnext_ref.T @ nextAM.Q @ xnext_ref - xk.T @ curAM.A.T @ nextAM.Q @ xnext_ref -\
                        xnext_ref.T @ nextAM.Q @ curAM.A @ xk + xk.T @ curAM.A.T @ nextAM.Q @ curAM.A @ xk).item()

        return offset

    def local_search(self, x0: np.ndarray) -> Tuple[List, List, float]:
        """
        Performs local search for each time step.

        Parameters:
            x0 (np.ndarray): The initial state from which the local search starts.

        Returns:
            Tuple[List, List, float]: A tuple containing local results, state-action pairs, and local optimal cost.
        """

        local_results = []

        x = x0.reshape(-1,1)

        state_action_pair = []

        local_opt_cost = 0

        xs = []
        us = []

        for k, action_model in enumerate(self.action_models[:-1]):

            next_action_model = self.action_models[k+1]

            # Create a model instance
            model = Model('Single-Step TO')

            u_binary = model.binary_var_matrix(action_model.nu,action_model.M)

            u = []
            for i in range(action_model.nu):
                val = model.sum(action_model.u_min)
                for j in range(action_model.M):

                    u_element = u_binary[(int(i),int(j))]
                    val += u_element * 2**j * ((action_model.u_max - action_model.u_min) / (2**action_model.M - 1))

                u.append(val)

            # Calculate the Cost Offset term
            cost_offset = self.calc_cost_offset(action_model, next_action_model, x)

            # Calculate the Decision-Variable dependent cost terms
            Rk = action_model.R.copy()
            xk_ref = action_model.x_ref.reshape(action_model.nx,1)
            uk_ref = action_model.u_ref.reshape(action_model.nu,1)
            xnext_ref = next_action_model.x_ref.reshape(next_action_model.nx,1)
            Ak = action_model.A.copy()
            Bk = action_model.B.copy()

            # Current Timestep's u-dependent Cost Terms
            cost_u = 0

            for i in range(action_model.nu):
                for j in range(action_model.nu):

                    # + uk.T @ Rk @ uk
                    cost_u += Rk[i][j].item() * u[i] * u[j]

                    # # - uk.T @ Rk @ uk_ref - uk_ref.T @ Rk @ uk.T
                    cost_u += -(uk_ref.T @ Rk).reshape(action_model.nu,)[i].item() * u[i]
                    cost_u += -(Rk @ uk_ref).reshape(action_model.nu,)[i].item() * u[i]

                    # Next Timestep's u-dependent Cost Terms
                    Qnext = next_action_model.Q.copy()

                    # + uk.T @ (Bk.T @ Qk+1 @ Bk) @ uk
                    cost_u += (Bk.T @ Qnext @ Bk)[i][j].item() * u[i] * u[j]

                    # - uk.T @ B.T @ Qk+1 @ xk+1_ref - xk+1_ref.T @ Qk+1 @ B @ uk
                    cost_u += -(Bk.T @ Qnext @ xnext_ref).reshape(action_model.nu,)[i].item() * u[i]
                    cost_u += -(xnext_ref.T @ Qnext @ Bk).reshape(action_model.nu,)[i].item() * u[i]

                    # + (xk.T @ Ak.T @ Qk+1 @ B) @ uk + uk.T @ (Bk.T @ Qk+1 @ A @ xk)
                    cost_u += (x.T @ Ak.T @ Qnext @ Bk).reshape(action_model.nu,)[i].item() * u[i]
                    cost_u += (Bk.T @ Qnext @ Ak @ x).reshape(action_model.nu,)[i].item() * u[i]

            # Assemble the Total Cost Function [Cost(u) + Offset]
            objective = cost_u + cost_offset

            # Define minimization problem
            model.minimize(objective)

            # Convert DOCPlex to QP
            qp = QuadraticProgram()
            qp.from_docplex(model)


            noise_model = NoiseModel()

            # Parameters
            prob_1 = 0  # 1-qubit gate error probability
            prob_2 = 0  # 2-qubit gate error probability

            # Add depolarizing error to single-qubit gates
            error_1 = noise.depolarizing_error(prob_1, 1)
            noise_model.add_all_qubit_quantum_error(error_1, ['x', 'h'])

            # Add depolarizing error to two-qubit gates
            error_2 = noise.depolarizing_error(prob_2, 2)
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

            # Create a noisy backend
            noisy_backend = Aer.get_backend('qasm_simulator')
            quantum_noise_instance = QuantumInstance(noisy_backend, shots=1024, noise_model=noise_model)

            # Create the GroverOptimizer with the noisy sampler
            noisy_grover_optimizer = GroverOptimizer(action_model.M, quantum_instance=quantum_noise_instance)

            # Solve with GroverOptimiser
            backend = Aer.get_backend('qasm_simulator')
            quantum_instance = QuantumInstance(backend, shots=1024)
            grover_optimizer = GroverOptimizer(action_model.M, quantum_instance=quantum_instance)
            result = noisy_grover_optimizer.solve(qp)
            local_results.append(result)

            u_gr = action_model.u_min + sum(result.x[j] * 2**j *\
                    ((action_model.u_max - action_model.u_min) / (2**action_model.M - 1)) for j in range(action_model.M))

            if self.verbose:
                print(f"\nGrover Result:\n{result}")
                print(f"Optimal u: {u_gr}\n")

            # Integrate the Dynamics xnext = f(xk, uk*)
            xnext = action_model.calc_dynamics(x, u_gr)

            if self.debug:

                # Solve with Eigen Solver (Exact Solver)
                exact_result = self.exact_solver.solve(qp)
                print(f"Exact Result:\n {exact_result}" )
                u_eig = action_model.u_min + sum(result.x[j] * 2**j *\
                        ((action_model.u_max - action_model.u_min) / (2**action_model.M - 1)) for j in range(action_model.M))
                print(f"Optimal u_eig: {u_eig}\n")
                xnexteig = action_model.calc_dynamics(x,u_eig)

                np.testing.assert_almost_equal(u_gr, u_eig, 3)
                np.testing.assert_almost_equal(xnext, xnexteig, 3)
                print(f"Timestep {k}: Test Successful")

            if self.verbose:
                print(f"Timestep {k} | x = {x.flatten()} | u = {u_gr} | xnext ={xnext} | cost = {result.fval}\n-----------------------\n")

            xs.append(x.flatten())
            us.append(u_gr)

            local_opt_cost += action_model.calc_cost(x.flatten(), u_gr.flatten(), action_model.Q, action_model.R)

            x = xnext.reshape(next_action_model.nx,1)

        xs.append(x.flatten())
        us.append(self.action_models[-1].u_none)

        local_opt_cost += self.action_models[-1].calc_cost(x.flatten(), self.action_models[-1].u_none, self.action_models[-1].Q, self.action_models[-1].R)

        return local_results, xs, us, local_opt_cost


    # def calculate_local_cost(self,xs, us) -> None:

    #     local_cost = 0.0

    #     for i, action_model in enumerate(self.action_models[:-1]):

    #         local_cost += action_model.calc_local_cost(xs[i], us[i])

    #     local_cost += self.action_models[-1].calc_local_term_cost(xs[-1], self.action_models[-1].u_none)

    #     return local_cost


    def calculate_qubits_needed(self, value: int) -> int:
        """
        Calculates the number of qubits needed to represent a given value.

        Parameters:
            value (int): The value for which to calculate the number of qubits.

        Returns:
            int: The number of qubits needed.
        """
        return int(np.ceil(np.log2(value + 1)))

    def process_node(self, args):

        # print(f"Thread: {x}")
        node, u, action_model, x_min, x_max, N = args
        x_next = action_model.calc_dynamics(node.state, u)

        local_cost = action_model.calc_cost(node.state, u, action_model.Q, action_model.R)

        if np.all(x_min <= x_next) and np.all(x_next <= x_max):
            # It should have been action_model_next.[min, max] but it's the same for now
            # x_next_closest = self.find_closest_node(x_next, x_min, x_max, N)
            total_cost = node.cost + local_cost
            child_node = StateNode(x_next, u, node, total_cost)
            node.children.append(child_node)
            return child_node
        else:
            # print(f"xnext out of bounds: {x_next}")
            return None

    def find_closest_node(self, x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, N: int)->np.ndarray:

        num_values = 2 ** N
        step_size = (x_max - x_min) / (num_values - 1)

        # Using np.clip to clamp each element of x within the range [x_min, x_max]
        x_clamped = np.clip(x, x_min, x_max)

        # Calculate the steps for each element
        steps = np.round((x_clamped - x_min) / step_size)

        # Compute the discrete values for each element
        x_discrete = x_min + steps * step_size

        return x_discrete


    def build_parallelised_tree(self, am_list, x0: np.ndarray, Ns: int,  num_threads: int = 16) -> Tuple[StateNode, List[StateNode]]:

        root = StateNode(x0)
        current_level = [root]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for t, action_model in enumerate(am_list[:-1]):

                next_level = []
                task_args = []
                next_nodes = []

                if (t < Ns):
                    task_args = [(node, u, action_model, action_model.x_min, action_model.x_max, self.N_global)
                                for node in current_level
                                for u in np.arange(action_model.u_min, action_model.u_max, (action_model.u_max - action_model.u_min) / (2 ** self.M_global - 1))]

                else:
                    task_args = [(node, node.control_input, action_model, action_model.x_min, action_model.x_max, self.N_global)
                                        for node in current_level]

                next_nodes = {executor.submit(self.process_node, task_arg): task_arg for task_arg in task_args}

                for future in concurrent.futures.as_completed(next_nodes):

                    if future.result() is not None:
                        # print(f"{future.result()} is in")
                        next_level.append(future.result())

                print(f"Level {t+1} has {len(next_level)} nodes")
                current_level = next_level

        last_layer = current_level

        return last_layer

    # @functools.lru_cache(maxsize=None)
    def build_tree(self, x0: Tuple[float, float]) -> Tuple[StateNode, List[StateNode]]:
        """
        Builds a state space tree based on the provided parameters.

        Parameters:
            x_min (np.ndarray): Minimum bound for state.
            x_max (np.ndarray): Maximum bound for state.
            u_bin_list (np.ndarray): List of binary encoded control inputs.
            T (int): The total number of time steps.
            x0 (np.ndarray): The initial state.

        Returns:
            Tuple[StateNode, List[StateNode]]: The root node and the last layer of nodes in the tree.
        """
        root = StateNode(np.array(x0))
        current_level = [root]

        for t, action_model in enumerate(tqdm(self.action_models, desc="Timesteps", position=0)):
            next_level = []

            # Prepare ranges
            u_range = np.arange(action_model.u_min, action_model.u_max, (action_model.u_max - action_model.u_min) / (2 ** self.M_global - 1))
            x_min, x_max, N = action_model.x_min, action_model.x_max, action_model.N

            for node in tqdm(current_level, desc="Layer Nodes", position=1, leave=False):
                for u in u_range:
                    x_next = action_model.calc_dynamics(node.state, u)
                    local_cost = action_model.calc_cost(node.state, u)

                    if np.all(x_min <= x_next) and np.all(x_next <= x_max):
                        print(f"{node} is in")
                        total_cost = node.cost + local_cost
                        child_node = StateNode(x_next, u, node, total_cost)
                        node.children.append(child_node)
                        next_level.append(child_node)

            current_level = next_level

        return root, current_level

    def calc_sliding_window_cost(self, start_idx: int, xs: np.ndarray, us: np.ndarray) -> float:

        cost = 0

        for i, action_model in enumerate(self.action_models[start_idx:]):

            cost += action_model.calc_cost(xs[i], us[i], action_model.Q, action_model.R)

        return cost

    def solve_MPC(self, x0: np.ndarray, Ns: int, with_local: bool = True, noise_param: float = 0.0) -> List[np.ndarray]:

        if with_local:
            _, xs_local, us_local, cost_threshold = self.local_search(x0)
        else:
            cost_threshold = np.inf
            xs_local = []
            us_local = []

        x = x0.flatten()

        xs = [x0]
        us = []

        for t in tqdm(range(self.T - Ns)):

            # cost_threshold = self.calc_sliding_window_cost(t, xs_local, us_local)

            last_layer = self.build_parallelised_tree(self.action_models[t:], x, Ns, 200)

            # Create the Cost Buffer
            L = [node.total_cost for node in last_layer]

                #def findMin(self, values: List[float], num_iterations: int = 10, noise_param: float = 0.01, cost_threshold: float = 0.0) -> int:


            opt_idx, _ = self.qmin_algo.findMin(L, 50, noise_param, cost_threshold)

            print(f"opt_idx: {opt_idx}, with total_idxs: {len(last_layer)}")

            if opt_idx is None:

                print(f"Timestep {t} Optimal Trajectory: Local\n--------------")

                u = us_local[t]
                x = self.action_models[t].calc_dynamics(x, u)

                cost_threshold = self.calc_sliding_window_cost(t+1, xs_local, us_local)


            else:

                opt_node = last_layer[opt_idx]

                L_global = L[opt_idx]

                opt_traj = self.extract_opt_trajectory(opt_node)

                print(f"Timestep {t} Optimal Trajectory: {opt_idx} with {L_global}\n--------------")

                cost_threshold = L_global


                u = opt_traj[1].control_input
                x = self.action_models[t].calc_dynamics(x, u)

            xs.append(x)
            us.append(u)

        if Ns >= 1:
            for node in opt_traj[2:]:
                xs.append(node.state)
                us.append(node.control_input)

        return xs, us, xs_local, us_local


    def extract_opt_trajectory(self, node: StateNode) -> List[StateNode]:
        """
        Traces back the path from a given node to the root node in the state space tree.

        Parameters:
            node (StateNode): The node to start tracing back from.

        Returns:
            List[StateNode]: A list of StateNodes representing the path from the root node to the given node.
        """
        path = []
        current_node = node
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.parent

        # Optionally, reverse the path if you want it from root to the given node
        path.reverse()
        return path


if __name__ == "__main__":

    # Parameters for the system
    T = 60 # Number of time steps
    M_local = 4# Number of bits per control
    N = 10 # Number of bits per state [DEPRECATED]
    u_min, u_max = -10, 10
    LQR_R = 0.001  # Weight in the cost function
    dt = 0.05 # Time step size}
    LQR_Q = np.eye(4) # State cost matrix
    x0 = np.array([0, np.pi, 0, 0]).flatten() # Initial state
    xtarget = np.array([0, 0, 0, 0]).flatten() # Terminal state

    LQR_Q[0,0] = 10.0
    LQR_Q[1,1] = 0.001
    LQR_Q[3,3] = 0.001
    LQR_Q[3,3] = 0.001

    x_min = np.array([-2, -np.pi * 2, -10, -4 * np.pi])
    x_max = np.array([2, np.pi * 2, 10, 4 * np.pi])

    action_models = []

    # Running Action Models
    for i in range(T-1):

        if i < 3*T/4:
            M_global = 5
        else:
            M_global = 7

        xref = xtarget
        run_model = CartPoleActionModel(N, M_local, deepcopy(M_global), u_min, u_max, x_min, x_max, LQR_Q, LQR_R,dt, x_ref=xref, u_ref=0)
        run_model.ws = 100.0
        run_model.wc = 100.0
        action_models.append(deepcopy(run_model))



    LQR_Qterm = np.eye(4)
    LQR_Qterm[0,0] = 3000.0
    LQR_Qterm[1,1] = 0.01
    LQR_Qterm[3,3] = 200.0
    LQR_Qterm[3,3] = 0.01
    LQR_Rterm = 0.0001

    term_model = CartPoleActionModel(N, M_local, M_global, u_min, u_max, x_min, x_max, LQR_Qterm, LQR_Rterm,dt, x_ref =xtarget, u_ref=0)

    term_model.ws = 10000
    term_model.wc = 10000

    action_models.append(term_model)

    traj = {}
    noise_param = 0
    solver = GAS_DP_Solver(action_models, N, M_global, True, False)

    for k in range(40):

        xs = []
        us = []

        xs, us, xs_local, us_local = solver.solve_MPC(x0, 2, False, noise_param)

        print(f"MPC Iterations Completed: {k}")
        cost = 0

        for i, action_model in enumerate(action_models[:-1]):
            cost += action_model.calc_cost(xs[i], us[i], action_model.Q, action_model.R)

        cost += action_models[-1].calc_cost(xs[-1], np.zeros_like(us[-1]), action_models[-1].Q, action_models[-1].R)
        print(f"Cost: {cost}")

        traj[f'xs{k}'] = deepcopy(xs)
        traj[f'us{k}'] = deepcopy(us)
        traj[f'cost{k}'] = deepcopy(cost)
        traj[f'xs_local{k}'] = deepcopy(xs_local)
        traj[f'us_local{k}'] = deepcopy(us_local)

#    np.savez("/workspace/results/cartpole_resdata", **traj, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/workspace/results/cartpole_resdata.pkl", "wb") as f:
        pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Data Saved!")
