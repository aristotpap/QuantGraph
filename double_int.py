from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import GroverOptimizer, MinimumEigenOptimizer
from qiskit.optimization.converters import QuadraticProgramToQubo
from qiskit import Aer, execute


try:
    from qiskit.aqua import QuantumInstance
    from qiskit.aqua.algorithms import NumPyMinimumEigensolver
except ImportError:
    try:
        from qiskit.utils import QuantumInstance
        from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
    except ImportError:
        from qiskit.utils import QuantumInstance
        from qiskit.algorithms import NumPyMinimumEigensolver


from docplex.mp.model import Model
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
import pickle

# Imports related to noise simulation
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise.errors import depolarizing_error


class DoubleIntegratorActionModel:
    """
    A model for a double integrator action system. This class provides methods
    to calculate the dynamics and cost for a given state and control input,
    as well as methods for binary encoding of control inputs.

    Attributes:
        N (int): Number of bits for state (x) binary encoding.
        M (int): Number of bits for control-input (u) binary encoding.
        u_min (float): Minimum control input.
        u_max (float): Maximum control input.
        Q (numpy.ndarray): State cost matrix.
        R (float): Control cost.
        dt (float): Time step for the dynamics.
        x_ref (numpy.ndarray): Reference state.
        u_ref (float): Reference control input.
    """

    def __init__(self, N: int, M: int, u_min: float, u_max: float,
                 x_min: np.ndarray, x_max: np.ndarray,
                 Q: np.ndarray = np.eye(2), R: float = 1, dt: float = 0.01,
                 x_ref: np.ndarray = np.array([0,0]), u_ref: float = 0):
        """
        Constructs all the necessary attributes for the DoubleIntegratorActionModel object.

        Parameters:
            N (int): Number of bits for state (x) binary encoding.
            M (int): Number of bits for control-input (u) binary encoding.
            u_min (float): Minimum control input.
            u_max (float): Maximum control input.
            Q (numpy.ndarray): State cost matrix. Defaults to 2x2 identity matrix.
            R (float): Control cost. Defaults to 1.
            dt (float): Time step for the dynamics. Defaults to 0.01.
            x_ref (numpy.ndarray): Reference state. Defaults to array [0, 0].
            u_ref (float): Reference control input. Defaults to 0.
        """

        self.nx = 2
        self.nu = 1

        self.u_none = np.zeros((self.nu,))

        self.N = N
        self.M = M
        self.u_min = u_min
        self.u_max = u_max

        self.x_ref = np.array(x_ref).reshape(self.nx,)

        self.u_ref = np.array(u_ref).reshape(self.nu,)

        self.x_min = x_min.flatten()
        self.x_max = x_max.flatten()

        self.Q = np.array(Q).reshape(self.nx, self.nx)
        self.R = np.array(R).reshape(self.nu, self.nu)

        self.dt = dt

        self.A = np.array([[1, self.dt], [0, 1]])
        self.B = np.array([0.5 * self.dt**2, self.dt]).reshape(2,1)

        pass

    def calc_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculates the next state of the system using double integrator dynamics.

        Parameters:
            x (numpy.ndarray): Current state of the system.
            u (float): Current control input.

        Returns:
            numpy.ndarray: The next state of the system.
        """

        u = np.array(u).reshape(self.nu, 1)
        x_next = (self.A @ x.reshape(self.nx,1) + self.B @ u).flatten()

        return x_next

    def calc_cost(self, x: np.ndarray, u: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
        """
        Calculates the cost based on the current state, control input, and their respective references.

        Parameters:
            x (numpy.ndarray): Current state of the system.
            u (float): Current control input.

        Returns:
            float: The calculated cost.
        """

        dx = (self.x_ref - x.flatten()).reshape(self.nx,1)

        du = (self.u_ref - np.array(u).reshape(self.nu,))

        cost_x = (dx.T @ Q @ dx).item()
        cost_u =  (du.T @ R @ du).item()

        return cost_x + cost_u

    # def calc_local_cost(self, x: np.ndarray, u: np.ndarray) -> float:

    #     Q = np.eye(self.nx)
    #     R = np.eye(self.nu)

    #     Q[0,0] = 10
    #     Q[1,1] = 0.001

    #     R = np.array([0.001])

    #     cost = self.calc_cost(x, u, Q, R)

    #     return cost

    def calc_local_term_cost(self, x: np.ndarray, u: np.ndarray) -> float:

        Q = np.eye(self.nx) * 100.0

        R = np.array([0.1])

        term_cost = self.calc_cost(x, u, Q, R)

        return term_cost

class GAS_DP_Solver:
    """
    A solver class implementing the QuantGraph algorithm using implicit QUBO formulation
    for both local (DP-based warm start) and global (MPC) search strategies.
    """

    def __init__(self, action_model_list, N_global: int, M_global: int, verbose: bool = False, debug: bool = False):
        self.action_models = action_model_list
        self.T = len(self.action_models)

        self.debug = debug
        self.verbose = verbose

        if self.debug:
            self.exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        else:
            self.exact_solver = None

        self.N_global = N_global # Unused
        self.M_global = M_global # Precision for Global Stage

        print(f"Initialized GAS_DP_Solver (Implicit QUBO) with {self.T} action models.")
        pass

    def calc_cost_offset_local(self, curAM, nextAM, x: np.ndarray) -> float:
        # Calculates the constant terms (independent of u_k) for the local search optimization.
        # This helper reconstructs the specific cost formulation used in the original implementation's local_search.

        offset = 0.0
        xk = x.reshape(curAM.nx,1)

        xk_ref = curAM.x_ref.reshape(curAM.nx,1)
        uk_ref = curAM.u_ref.reshape(curAM.nu,1)
        xnext_ref = nextAM.x_ref.reshape(nextAM.nx,1)

        # Using (xref - x) as in the original implementation's local search derivation
        dx_k = (xk_ref - xk).copy()

        # Current Timestep's Offset
        offset += (dx_k.T @ curAM.Q @ dx_k + uk_ref.T @ curAM.R @ uk_ref).item()

        # Next Timestep's Offset Contribution
        offset += (xnext_ref.T @ nextAM.Q @ xnext_ref - xk.T @ curAM.A.T @ nextAM.Q @ xnext_ref -\
                        xnext_ref.T @ nextAM.Q @ curAM.A @ xk + xk.T @ curAM.A.T @ nextAM.Q @ curAM.A @ xk).item()

        return offset

    def local_search(self, x0: np.ndarray) -> Tuple[List, List, float]:
        """
        Performs local search (one step lookahead optimization) for each time step.
        (This implementation remains largely unchanged as it already used implicit QUBO via GroverOptimizer)
        """

        local_results = []
        x = x0.reshape(-1,1)
        local_opt_cost = 0
        xs = []
        us = []

        # Setup ideal QuantumInstance for local search (as in original implementation)
        backend = Aer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024)

        for k, action_model in enumerate(self.action_models[:-1]):

            next_action_model = self.action_models[k+1]
            M_loc = action_model.M # Precision defined in the model initialization (M_local)

            # Create a model instance
            model = Model('Single-Step TO')

            u_binary = model.binary_var_matrix(action_model.nu, M_loc)

            # Construct symbolic continuous control u_k
            u = []
            delta = (action_model.u_max - action_model.u_min) / (2**M_loc - 1)
            for i in range(action_model.nu):
                val = model.sum(action_model.u_min)
                for j in range(M_loc):
                    u_element = u_binary[(int(i),int(j))]
                    val += u_element * (2**j) * delta
                u.append(val)

            # Calculate the Cost Offset term
            cost_offset = self.calc_cost_offset_local(action_model, next_action_model, x)

            Rk = action_model.R.copy()
            uk_ref = action_model.u_ref.reshape(action_model.nu,1)
            xnext_ref = next_action_model.x_ref.reshape(next_action_model.nx,1)
            Ak = action_model.A.copy()
            Bk = action_model.B.copy()
            Qnext = next_action_model.Q.copy()

            cost_u = 0

            for i in range(action_model.nu):
                for j in range(action_model.nu):
                    # Quadratic terms
                    cost_u += Rk[i][j].item() * u[i] * u[j]
                    cost_u += (Bk.T @ Qnext @ Bk)[i][j].item() * u[i] * u[j]

                    # Linear terms
                    cost_u += -(uk_ref.T @ Rk).reshape(action_model.nu,)[i].item() * u[i]
                    cost_u += -(Rk @ uk_ref).reshape(action_model.nu,)[i].item() * u[i]
                    cost_u += -(Bk.T @ Qnext @ xnext_ref).reshape(action_model.nu,)[i].item() * u[i]
                    cost_u += -(xnext_ref.T @ Qnext @ Bk).reshape(action_model.nu,)[i].item() * u[i]
                    cost_u += (x.T @ Ak.T @ Qnext @ Bk).reshape(action_model.nu,)[i].item() * u[i]
                    cost_u += (Bk.T @ Qnext @ Ak @ x).reshape(action_model.nu,)[i].item() * u[i]


            # Assemble the Total Cost Function
            objective = cost_u + cost_offset
            model.minimize(objective)

            # Convert DOCPlex to QP
            qp = QuadraticProgram()
            qp.from_docplex(model)

            # Solve with GroverOptimizer
            grover_optimizer = GroverOptimizer(M_loc, quantum_instance=quantum_instance)
            result = grover_optimizer.solve(qp)
            local_results.append(result)

            # Decode the result
            u_gr = action_model.u_min + sum(result.x[j] * (2**j) * delta for j in range(M_loc))

            # Integrate the Dynamics
            xnext = action_model.calc_dynamics(x, u_gr)

            if self.verbose:
                print(f"Local Search Timestep {k} | u = {u_gr} | cost = {result.fval}")

            xs.append(x.flatten())
            us.append(u_gr)

          
            local_opt_cost += action_model.calc_cost(x.flatten(), u_gr.flatten(), action_model.Q, action_model.R)

            x = xnext.reshape(action_model.nx,1)

        # Handle terminal state and cost
        xs.append(x.flatten())
        us.append(self.action_models[-1].u_none)
        local_opt_cost += self.action_models[-1].calc_cost(x.flatten(), self.action_models[-1].u_none, self.action_models[-1].Q, self.action_models[-1].R)

        return local_results, xs, us, local_opt_cost

    # --- Global Stage Implementation (Implicit QUBO) ---

    def synthesize_Ns_QUBO(self, t_start: int, x0: np.ndarray, Ns: int, penalty_factor=1e6):
        T = self.T
        # Ensure the horizon is within bounds
        if t_start + Ns > T:
            Ns = T - t_start

        if Ns == 0:
            return None, None

        model = Model(f'Ns-Step TO (t={t_start}, Ns={Ns})')
        M = self.M_global # Use global precision

        u_binary = {}
        U_symbolic = {}

        for k in range(Ns):
            action_model = self.action_models[t_start + k]
            
            # Assuming nu=1 for Double Integrator
            if action_model.nu != 1:
                 raise NotImplementedError("QUBO synthesis currently only supports systems with nu=1")

            u_binary[k] = []
            for j in range(M):
                u_binary[k].append(model.binary_var(name=f"b_{t_start+k}_{j}"))

            # Construct Symbolic Continuous Control u_k
            u_min = action_model.u_min
            u_max = action_model.u_max
            delta = (u_max - u_min) / (2**M - 1)

            # DOCPLEX handles the symbolic summation
            u_k = model.sum(u_min)
            for j in range(M):
                u_k += u_binary[k][j] * (2**j) * delta

            U_symbolic[k] = u_k

        X_symbolic = {}
        X_symbolic[0] = x0.flatten().tolist()

        total_cost = 0

        for k in range(Ns):
            action_model = self.action_models[t_start + k]
            A, B, Q, R = action_model.A, action_model.B, action_model.Q, action_model.R
            x_ref, u_ref = action_model.x_ref.flatten(), action_model.u_ref.flatten()

            x_k = X_symbolic[k]
            u_k = U_symbolic[k]

           
            for i in range(action_model.nx):
                for j in range(action_model.nx):
                    dx_i = x_k[i] - x_ref[i]
                    dx_j = x_k[j] - x_ref[j]
                    total_cost += dx_i * Q[i,j] * dx_j

            du = u_k - u_ref[0]
            total_cost += du * R[0,0] * du

            x_next = []
            for i in range(action_model.nx):
                Ax = sum(A[i, j] * x_k[j] for j in range(action_model.nx))
                Bu = B[i, 0] * u_k
                x_next.append(Ax + Bu)

            X_symbolic[k+1] = x_next

            
            # for i in range(action_model.nx):
            #      model.add_constraint(x_next[i] >= action_model.x_min[i])
            #      model.add_constraint(x_next[i] <= action_model.x_max[i])

        model.minimize(total_cost)

        qp_constrained = QuadraticProgram()
        qp_constrained.from_docplex(model)

  
        converter = QuadraticProgramToQubo(penalty=penalty_factor)
        qp_qubo = converter.convert(qp_constrained)

        return qp_qubo, converter

    def solve_MPC(self, x0: np.ndarray, Ns: int, with_local: bool = True, noise_param: float = 0.0) -> List[np.ndarray]:
      

        # Local search (Warm start)
        if with_local:
            if self.verbose: print("Starting Local Search...")
            _, xs_local, us_local, _ = self.local_search(x0)
        else:
            xs_local, us_local = [], []

        x = x0.flatten()
        xs = [x0]
        us = []
        T = self.T
        M_g = self.M_global

        # Setup Quantum Instance and Noise Model for Global Stage
        noise_model = NoiseModel()
        if noise_param > 0:
            # Define a noise model based on the noise_param
            # Heuristic scaling for 1-qubit and 2-qubit gate errors
            prob_1 = noise_param / 10
            prob_2 = noise_param

            error_1 = noise.depolarizing_error(prob_1, 1)
            # Including standard gates often used in synthesis
            noise_model.add_all_qubit_quantum_error(error_1, ['x', 'h', 'u1', 'u2', 'u3', 'id', 'rz', 'sx'])

            error_2 = noise.depolarizing_error(prob_2, 2)
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

        backend = Aer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024, noise_model=noise_model if noise_param > 0 else None)


        # MPC Loop
        if self.verbose: print("Starting Global Stage (MPC Loop)...")
        for t in tqdm(range(T)):
            # Determine the effective horizon (shrinks near the end)
            current_Ns = min(Ns, T - t)
            if current_Ns == 0:
                break

          
            qp_qubo, converter = self.synthesize_Ns_QUBO(t, x, current_Ns, penalty_factor=1e6)

            if qp_qubo is None:
                break

          
            num_qubits = qp_qubo.get_num_vars()
            
            grover_optimizer = GroverOptimizer(num_qubits, quantum_instance=quantum_instance)

            # Solve
            result_qubo = grover_optimizer.solve(qp_qubo)

     
            # Build a name->index map for the QUBO variables in the SAME order as result_qubo.x
            name_to_index = {v.name: i for i, v in enumerate(qp_qubo.variables)}

            action_model = self.action_models[t]
            bits_u_t = []
            missing = False
            for j in range(M_g):
                var_name = f"b_{t}_{j}"
                idx = name_to_index.get(var_name)
                if idx is None:
                    missing = True
                    break
                # Grover returns values very close to 0/1; round to be safe
                bits_u_t.append(int(round(float(result_qubo.x[idx]))))

            if missing:
                print(f"Warning: Variable {var_name} not found in QUBO solution; falling back.")
                if with_local and t < len(us_local):
                    u_gr = us_local[t]
                else:
                    u_gr = action_model.u_ref.flatten()
            else:
                delta = (action_model.u_max - action_model.u_min) / (2**M_g - 1)
                u_gr = action_model.u_min + sum(bits_u_t[j] * (2**j) * delta for j in range(M_g))
                if self.verbose:
                    print(f"\nTimestep {t} (Ns={current_Ns}, Qubits={num_qubits}) | Global Grover Result Cost: {result_qubo.fval:.2f}")
                    print(f"Applied u_t: {u_gr}")

            # 4. Apply control and advance dynamics
            x_next = action_model.calc_dynamics(x, u_gr)

            # Update state and record trajectory
            x = x_next.flatten()
            xs.append(x)
            us.append(u_gr)

        return xs, us, xs_local, us_local

if __name__ == "__main__":

    T = 12 # Number of time steps
    M_local = 10  # Number of bits per control
    M_global = 6
    N = 10 # Number of bits per state [DEPRECATED]
    u_min, u_max = -5, 5
    LQR_R = 0.1  # Weight in the cost function
    dt = 0.1 # Time step size}
    LQR_Q = np.eye(2) # State cost matrix
    x0 = np.array([0, 0]).flatten() # Initial state
    xtarget = np.array([1, 0]).flatten() # Terminal state

    LQR_Q[0, 0] = 2000
    LQR_Q[1, 1] = 50


    xref = x0

    x_min = np.array([-2.0, -2.0])
    x_max = np.array([2.0, 2.0])


    action_models = []
    # Running Action Models

    dx_step = (xtarget - x0) / T
    for i in range(T-1):

        xref = xtarget
        run_model = DoubleIntegratorActionModel(N, M_local, u_min, u_max, x_min, x_max, LQR_Q, LQR_R,dt, x_ref=xref, u_ref=0)
        action_models.append(run_model)

    LQR_Qterm = np.eye(2) * 1e8
    term_model = DoubleIntegratorActionModel(N, M_local, u_min, u_max, x_min, x_max, LQR_Qterm, LQR_R,dt, x_ref =xtarget, u_ref=0)

    action_models.append(term_model)

    traj = {}
    noise_param = 0
    solver = GAS_DP_Solver(action_models, N, M_global, True, False)

    for k in range(20):

        xs = []
        us = []

        xs, us, xs_local, us_local = solver.solve_MPC(x0, 2, True, noise_param)

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

    #np.savez("/workspace/results/double_integrator_resdata", **traj, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/workspace/results_implicit/double_integrator_resdata.pkl", "wb") as f:
        pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Data Saved!")
