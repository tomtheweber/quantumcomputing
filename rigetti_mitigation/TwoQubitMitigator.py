# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:13:06 2020

@author: Hando
"""

import numpy as np
import matplotlib.pyplot as plt


from pyquil import get_qc
from pyquil.quil import Program
from pyquil.gates import RX, RZ, X, CNOT, H, S

from pyquil.paulis import sZ, PauliSum, PauliTerm, commuting_sets
from pyquil.api import WavefunctionSimulator, QuantumComputer


from typing import Union, Optional, List, Dict, Tuple
import time, warnings


class TwoQubitMitigator:
    """Class used to compute readout error-mitigated expectation values of two-qubit circuits.
    
    
    
    Attributes:
        operator: A PauliSum or a PauliTerm to be evaluated on a state given by a quantum circuit.
        
        device: A QuantumComputer object on which we like to run our experiments.
        
        num_shots_calibration: An integer representing the number of experiments run for each calibration circuit.
        
        num_shots_evaluation: An integer representing the number of experiments run for the actual evaluation circuits.  
        
        noisy_readout_probabilities: A dictionary containing artificially implemented bit-flip probabilities for the qubits.
                                     Each value of the Dict is a tuple (p00,p11), where 1-p00 is the probability of a flip 0->1.
    """
    
    
    def __init__(self,
                 operator: Union[PauliSum, PauliTerm],
                 device: QuantumComputer,
                 num_shots_calibration: Optional[int] = 8192,
                 num_shots_evaluation:  Optional[int] = 8192,
                 noisy_readout_probabilities: Optional[Dict[int,Tuple[float, float]]]={}
                 ) ->None:
        
       
            
        self._operator=operator
        self._device=device
        self._num_shots_calibration=num_shots_calibration
        self._num_shots_evaluation=num_shots_evaluation
        self._noisy_readout_probabilities=noisy_readout_probabilities
        
        
        # create a list of the single PauliTerm objects of the given operator
        if isinstance(self._operator, PauliSum):
            self._terms=self._operator.terms
        
        else: self._terms=[self._operator]
            
        
        # all qubits affected by the operator
        self._qubits=self._operator.get_qubits()
        
        if len(self._qubits)>2:
            raise ValueError('The number of qubits of the passed operator is larger than 2. Right now, the mitigation is only implemented for two qubits.')
        

        
    def __repr__(self):
        return 'TwoQubitMitigator(operator: {0}, device: {1}, shots: {2},{3}'.format(self._operator,self._device, 
                                                                                     self._num_shots_calibration,
                                                                                     self._num_shots_evaluation)
        
    def calibration_circuits(self,
                             allow_multi_qubit: Optional[bool] = True
                             )->Dict[int,Program]:
        """Create a list of Program objects for calibrating.
        
        Args:
            allow_multi_qubit: A bool indicating whether we want to calibrate both qubits simultaneously.
            
        Returns:
            A list of Program objects for the calibration experiments.
        """
        circuits=dict()
        
        
        
        # in case we want to calibrate both qubits simultaneously
        if allow_multi_qubit:
            # start with calibration of |0>
            prog = Program('PRAGMA INITIAL_REWIRING "NAIVE"')
            ro = prog.declare('ro', 'BIT', 2)
            
            for index, qubit in enumerate(sorted(self._qubits)):                
                prog += Program().measure(qubit, ro[index])
            circuits['0']=prog
            
            # continue with |1>
            prog = Program('PRAGMA INITIAL_REWIRING "NAIVE"')
            ro = prog.declare('ro', 'BIT', 2)
            
            for index, qubit in enumerate(sorted(self._qubits)):  
                prog += X(qubit)
                prog += Program().measure(qubit, ro[index])
            

            circuits['1']=prog
        
        # otherwise
        else:
            
            # start with |0> state for both qubits separately
            for index, qubit in enumerate(self._qubits):
                prog=Program('PRAGMA INITIAL_REWIRING "NAIVE"')
                ro = prog.declare('ro', 'BIT', 1)
                prog += Program().measure(qubit, ro[0])
                circuits[str(qubit)+'0']=prog
            # continue with |1>, again separately for the two qubits
            for index, qubit in enumerate(self._qubits):
                prog=Program('PRAGMA INITIAL_REWIRING "NAIVE"')
                ro = prog.declare('ro', 'BIT', 1)
                prog += X(qubit)
                prog += Program().measure(qubit, ro[0])
                circuits[str(qubit)+'1']=prog

        return circuits
    
    
    def _evaluate_single_term(self,
                              ansatz_circuit: Program,
                              meas_term: PauliTerm
                              ) -> np.ndarray:
        """Compute the expectation value of a single PauliTerm , given a quantum circuit to evaluate on.
        
        Args:
            ansatz_circuit: A Program representing the quantum state to evaluate the PauliTerm on.
            
            meas_term: a PauliTerm object to be evaluated.
        
        """
        # First, create the quantum circuit needed for evaluation.
        
        # concatenate operator (converted to a Program) and ansatz circuit
        expectation_circuit=ansatz_circuit.copy()
        
        
        expectation_circuit+=meas_term.program
            
        ro = expectation_circuit.declare('ro', 'BIT', len(meas_term.get_qubits()))
        
        
        # add necessary post-rotations and measurement
        for i,qubit in enumerate(sorted(list(meas_term.get_qubits()))):
            
            if meas_term.pauli_string([qubit])=='X':
                expectation_circuit+=H(qubit)
            elif meas_term.pauli_string([qubit])=='Y':
                expectation_circuit+=H(qubit)
                expectation_circuit+=S(qubit)
                
            expectation_circuit+=Program().measure(qubit, ro[i])
            
        result=self._run(expectation_circuit, num_shots=self._num_shots_evaluation)
        
                
        
        
        return result
    

    def expectation_value(self,
                          ansatz_circuit: Program,
                          ) -> Tuple[float, float]:
        """Compute the mitigated as well as the non-mitigated expectation value of the operator on a given ansatz circuit.
        
        Args:
            ansatz_circuit: A Program representing the quantum state in which we want to evaluate the operator.
            
        Returns: 
            A tuple of the mitigated and the non-mitigated expectation value.
        
        """
        
        probabilities=self.get_bit_flip_probabilities(True)
        
        qubits=self.get_qubits()
        
        p0_first=probabilities[str(qubits[0])+'0']
        p1_first=probabilities[str(qubits[0])+'1']
        p0_second=probabilities[str(qubits[1])+'0']
        p1_second=probabilities[str(qubits[1])+'1']
        
        # compute the gammas according to the paper.
        gamma_Z1=1-p0_first-p1_first
        gamma_Z2=1-p0_second-p1_second
        gamma_I1=p1_first-p0_first
        gamma_I2=p1_second-p0_second
        
        # create expectation values and set them to zero
        expectation_value_mitigated=0.
        expectation_value_without=0.
        
        
        
        terms=self._terms
        
        for term in terms:
            result_first_qubit, result_second_qubit=self._evaluate_single_term(ansatz_circuit, term)
            
        return p1_first
            
        
        

                
                
            
        
            
        
            
        
        
    
    
    def get_bit_flip_probabilities(self,
                                   allow_multi_qubit: bool = True
                                   ) -> Dict[str,float]:
        """Compute the bit-flip probabilities on the relevant qubits for the mitigation.
        
        Args: 
            allow_multi_qubit: A bool indicating whether we calibrate all qubits simultaneously.
        
        Returns: A dictionary of lists of the form [p_q0, p_q1], where p_q0 and p_q1 denote the bit-flip
                 probabilities on the qubit with index q.
        """
        calibration_circuits=self.calibration_circuits(allow_multi_qubit)
        
        probabilities=dict()
        
        if allow_multi_qubit:
            
            # run circuit for |0> state and grab results
            result_0=self._run(calibration_circuits['0'] , self._num_shots_calibration).T
            
            # write probabilities into a dictionary with keys 'q0' and 'q1', where the entries denote the probability
            # of a bit-flip during readout of qubit q in state 0 or 1, respectively.
            for index, qubit in enumerate(self._qubits):
                probabilities[str(qubit)+'0']=sum(result_0[index])/len(result_0[index])
                
                
            # repeat the same for the |1> state
            result_1=self._run(calibration_circuits['1'] , self._num_shots_calibration).T
            
            for index, qubit in enumerate(self._qubits):
                probabilities[str(qubit)+'1']=(len(result_1[index])-sum(result_1[index]))/len(result_1[index])
            
                
            
        else:
            pass
        
        
        return probabilities
            

   
    
    def get_qubits(self):
        """Give a list of all qubits involved in the mitigation procedure.
        
        Returns: A list of qubit indices
        """

        return sorted(self._qubits)
    

    
    def _run(self,
             circuit: Program,
             num_shots: int=8192
             ) -> List[np.ndarray]:
        """Run a Program a number of times and record the measurement results.
        
        Args:
            circuit: A Program or a Python list of Program objects to be run and measured.
            num_shots: An integer representing the number of times the quantum circuit is to be evaluated.
            
        Returns: A list of numpy arrays containing the readout of qubits from single measurements.
        
        """
        prog=circuit.copy()
        
        #check whether we are given any bit-flip probabilities
        if self._noisy_readout_probabilities:
            
            # Add readout error to each qubit according to the dictionary
            for qubit in circuit.get_qubits(True):
                
                # it might happen that there are qubits without any probabilities given which would raise a KeyError
                try:
                    p0,p1=self._noisy_readout_probabilities[qubit]
                    prog.define_noisy_readout(qubit, p00=p0, p11=p1)
                except:
                    
                    # if we don't find the key in the dictionary, we don't add readout error the the qubit, but rather pass
                    pass
                
        # execute the circuit num_shots times    
        prog.wrap_in_numshots_loop(num_shots)
        result=self._device.run(self._device.compile(prog))

        return result
    