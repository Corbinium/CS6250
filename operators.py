import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

_NUM_QUBITS = 3
def set_num_qubits(num: int):
    """set_num_qubits(num: int) -> None:
        num: number of qubits to set for the global operators
        returns: None, but sets the global variables for the operators based on the number 
            of qubits
    """
    global _NUM_QUBITS, op_x, op_h, op_i, op_mcz, op_0, op_1
    _NUM_QUBITS = num
    op_x = tensor_series([OP_X] * _NUM_QUBITS)
    op_h = tensor_series([OP_H] * _NUM_QUBITS)
    op_i = tensor_series([OP_I] * _NUM_QUBITS)
    op_mcz = tensor_series([OP_I] * _NUM_QUBITS)
    op_mcz.matrix[-1, -1] = -1
    op_0 = tensor_series([Q_0] * _NUM_QUBITS)
    op_1 = tensor_series([Q_1] * _NUM_QUBITS)

class operator:
    """operator object
    Attributes:
        scaler: scalar multiplier for the operator
        matrix: matrix representation of the operator
    Methods:
        from_int(value: int) -> tuple[operator, operator]: creates operators from an integer, 
            the first operator represents the state of the qubits and the second operator 
            represents the state specific normalization operator
        operate(other: operator) -> operator: applies the operator to another operator and 
            returns the result
        state() -> str: returns the state represented by the operator as a binary string, this 
            will only work for single column vectors
        measure() -> dict[str, float]: returns a dictionary mapping basis states to their
            probabilities based on the amplitudes in the operator's matrix, this will only work 
            for single column vectors
        simulate_measurement() -> str: simulates a single measurement of the operator, returning 
            one basis state according to the probability distribution defined by the amplitudes 
            in the operator's matrix, this will only work for single column vectors
    """
    scaler: np.float64
    matrix: np.ndarray

    def __init__(self, scaler: np.float64, matrix: np.ndarray):
        self.scaler = scaler
        self.matrix = matrix

    @classmethod
    def from_int(cls, value: int) -> tuple['operator', 'operator']:
        state = f"{value:0{_NUM_QUBITS}b}"
        q_list = [Q_0 if bit == '0' else Q_1 for bit in state]
        op_q = tensor_series(q_list)
        a_list = [OP_X if bit == '0' else OP_I for bit in state]
        op_a = tensor_series(a_list)
        return op_q, op_a

    def __str__(self):
        return f"Scale: {self.scaler}\n{np.array2string(self.matrix.real, precision=2, suppress_small=False)}"

    def __repr__(self):
        return self.__str__()
    
    def operate(self, other: 'operator') -> 'operator':
        return operator(self.scaler * other.scaler, self.matrix @ other.matrix)
    
    def state(self) -> str:
        if self.matrix.shape[1] != 1:
            raise ValueError("State can only be determined for single column vectors.")
        index = np.argmax(np.abs(self.matrix))
        state = f"{index:0{_NUM_QUBITS}b}"
        return state
    
    def measure(self) -> dict[str, float]:
        if self.matrix.shape[1] != 1:
            raise ValueError("Measurement can only be performed on single column vectors.")
        probs = {}
        amplitudes = self.matrix.flatten() * self.scaler
        for i, amp in enumerate(amplitudes):
            prob = np.abs(amp) ** 2
            if prob > 1e-10:  # Only include non-zero probabilities
                state = f"{i:0{_NUM_QUBITS}b}"
                probs[state] = prob
        return probs

    def simulate_measurement(self) -> str:
        if self.matrix.shape[1] != 1:
            raise ValueError("Measurement can only be performed on single column vectors.")
        amplitudes = self.matrix.flatten() * self.scaler
        probs = np.abs(amplitudes) ** 2
        probs = probs / np.sum(probs)  # Normalize
        outcome = np.random.choice(len(probs), p=probs)
        return f"{outcome:0{_NUM_QUBITS}b}"

def tensor_series(ops: list[operator]) -> operator:
    """tensor_series(ops: list[operator]) -> operator
        ops: list of operators to be tensored together in the order to be applied
        returns: operator resulting from the tensor product of the input operators
    """
    result_matrix = ops[0].matrix
    for op in ops[1:]:
        result_matrix = np.kron(result_matrix, op.matrix)
    return operator(np.prod([op.scaler for op in ops]), result_matrix)

def operator_series(ops: list[operator]) -> operator:
    """operator_series(ops: list[operator]) -> operator:
        ops: list of operators to be applied in the order to apply them
        returns: operator resulting from the application of the input operators in sequence
    """
    result = ops[0]
    for op in ops[1:]:
        result = result.operate(op)
    return result

Q_0 = operator(1, np.array([[1], [0]]))
Q_1 = operator(1, np.array([[0], [1]]))

OP_X = operator(1, np.array([[0, 1], [1, 0]]))
OP_Z = operator(1, np.array([[1, 0], [0, -1]]))
OP_H = operator(1/np.sqrt(2), np.array([[1, 1], [1, -1]]))
OP_I = operator(1, np.array([[1, 0], [0, 1]]))

set_num_qubits(3)

def show_help():
    print(f"{'='*30}\n=== Single Qubit Operators")
    print("    Q_0, Q_1, OP_X, OP_Z, OP_H, OP_I\n")

    print(f"{'='*30}\n=== {_NUM_QUBITS}-Qubit Operators")
    print("    op_0, op_1, op_x, op_h, op_i, op_mcz\n")

    print(f"{'='*30}\n=== Functions")
    print(tensor_series.__doc__)
    print(operator_series.__doc__)
    print(set_num_qubits.__doc__)

    print(f"{'='*30}\n=== operator")
    print(operator.__doc__)

if __name__ == "__main__":
    show_help()
