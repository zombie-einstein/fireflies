import numpy as np


class Model:
    def __init__(self,
                 steps: int,
                 distances: np.array,
                 initial_phases: np.array,
                 epsilon: float,
                 b: float,
                 phase_step: int):
        """
        Initialize a new mode

        Args:
            steps (int): Number of steps to run the model for
            distances (np.array): 2d array distance matrix between nodes
            initial_phases (np.array): 1d array of initial node phases
            epsilon (float): epsilon parameter for response curve
            b (float): b value for response curve
            phase_step (int): Amount node phase increases each step
        """
        assert distances.ndim == 2
        assert distances.shape[0] == distances.shape[1]
        nodes = distances.shape[0]

        assert initial_phases.ndim == 1
        assert initial_phases.shape[0] == nodes

        horizon = int(np.max(distances))

        self.events = np.zeros((steps+horizon, nodes))
        self.distances = distances.astype("int")

        self.distances = self.distances[~np.eye(nodes, dtype=bool)].reshape(nodes, -1)
        self.node_idxs = np.array([np.arange(nodes) for _ in range(nodes)])
        self.node_idxs = self.node_idxs[~np.eye(nodes, dtype=bool)].reshape(nodes, -1)

        self.phases = np.zeros((steps, nodes))
        self.phases[0] = initial_phases
        self.steps = steps
        self.step = 0
        self.epsilon = epsilon
        self.b = b
        self.phase_step = phase_step
        self.threshold = 1

        self.alpha = np.exp(self.b * self.epsilon)
        self.beta = (self.alpha - 1) / (np.exp(self.b) - 1)

    def prc(self, phase: np.array):
        """Phase response function"""
        return np.minimum(phase * self.alpha + self.beta, self.threshold)

    def model_step(self):
        """
        One update step of the model

        - Update phases (taking into account the threshold value)
        - Create events for nodes that have fired
        - Advance time
        - Update phases dependent on the current events
        """
        pt = self.phases[self.step]
        self.phases[self.step] = np.where(pt >= self.threshold, 0, pt)
        pt = self.phases[self.step]

        for i in np.argwhere(pt == 0):
            i = i[0]
            futures = self.distances[i]+self.step
            idxs = self.node_idxs[i]
            self.events[futures, idxs] += 1

        self.step += 1

        et = self.events[self.step]

        self.phases[self.step] = np.where(et > 0,
                                          self.prc(pt),
                                          pt+self.phase_step)

    def run(self):
        """
        Run the model for the allotted number of steps
        """
        for _ in range(self.steps-1):
            self.model_step()
