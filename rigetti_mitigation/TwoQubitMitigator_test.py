# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:48:11 2020

@author: Tom
"""

from pyquil.paulis import sZ, sX, sY, PauliSum, PauliTerm, commuting_sets
from rigetti_mitigation.TwoQubitMitigator import TwoQubitMitigator
from pyquil import get_qc
from pyquil.quil import Program
from pyquil.gates import RX, RZ, X,Y,Z, CNOT, H
from typing import Dict
import tqdm
import time
import numpy as np
from pyquil.api import WavefunctionSimulator
import matplotlib.pyplot as plt

qpu_name = '8q-qvm' # 8q-qvm   Aspen-8
asqvm = True
isnoisy = False
QPU = get_qc(qpu_name,as_qvm=asqvm, noisy=isnoisy)
program_initialization = Program('PRAGMA INITIAL_REWIRING "NAIVE"')
operator=sZ(0)*sZ(1)
probabilities={}
#probabilities={0:(0.9,0.8)}
shots=2048
mitigator=TwoQubitMitigator(operator, device=QPU, num_shots_evaluation=shots, noisy_readout_probabilities=probabilities)


params={0: 2*np.pi*np.random.random(),
            1: 2*np.pi*np.random.random(),
            2: 2*np.pi*np.random.random(),
            3: 2*np.pi*np.random.random(),
            4: 2*np.pi*np.random.random(),
            5: 2*np.pi*np.random.random(),
            6: 2*np.pi*np.random.random(),
            7: 2*np.pi*np.random.random(),
            8: 2*np.pi*np.random.random(),
            9: 2*np.pi*np.random.random(),
            10: 2*np.pi*np.random.random(),
            11: 2*np.pi*np.random.random(),
            12: 2*np.pi*np.random.random(),
            13: 2*np.pi*np.random.random(),
            14: 2*np.pi*np.random.random()}

# initialise pyquil's simulator for calculation of exact expectation value
wv_sim=WavefunctionSimulator()

N=1024


def ansatz(params: Dict[int,float],
           prog_in: Program = program_initialization
           )-> Program:
    """
    Create a maximally expressive 2-qubit quantum circuit with minimal amount of parameters (15 rotations)
    
    Args:
        params: A dictionary of 15 real parameters for the 15 rotation gates in the ansatz circuit
        prog_in: A Program to start creating the circuit with.
        
    Returns: A Program representing the ansatz circuit.
    """
    prog_out = prog_in.copy()
    
    prog_out += RZ(params[0],  1)
    
    prog_out += CNOT(0,1)
    
    prog_out += RZ(params[1],  0)
    prog_out += RX(params[2],  0)
    prog_out += RZ(params[3],  0)
    
    prog_out += RZ(params[4],  1)
    prog_out += RX(params[5],  1)
    prog_out += RZ(params[6],  1)
    
    prog_out += CNOT(0,1)
    
    prog_out += RX(params[7],  0)
    prog_out += RZ(params[8],  1)
    
    prog_out += CNOT(0,1)
    
    prog_out += RZ(params[9],  0)
    prog_out += RX(params[10], 0)
    prog_out += RZ(params[11], 0)
    
    prog_out += RZ(params[12], 1)
    prog_out += RX(params[13], 1)
    prog_out += RZ(params[14], 1)


    return prog_out

true=float(np.real(wv_sim.expectation(prep_prog=ansatz(params), pauli_terms=[operator])))
true=np.zeros((shots,1))
estimate=np.zeros((shots,1))
not_mitigated=np.zeros((shots,1))

for i in tqdm.tqdm(range(N)):
    
    
    params={0: 2*np.pi*np.random.random(),
            1: 2*np.pi*np.random.random(),
            2: 2*np.pi*np.random.random(),
            3: 2*np.pi*np.random.random(),
            4: 2*np.pi*np.random.random(),
            5: 2*np.pi*np.random.random(),
            6: 2*np.pi*np.random.random(),
            7: 2*np.pi*np.random.random(),
            8: 2*np.pi*np.random.random(),
            9: 2*np.pi*np.random.random(),
            10: 2*np.pi*np.random.random(),
            11: 2*np.pi*np.random.random(),
            12: 2*np.pi*np.random.random(),
            13: 2*np.pi*np.random.random(),
            14: 2*np.pi*np.random.random()}

    
    true+=float(np.real(wv_sim.expectation(prep_prog=ansatz(params), pauli_terms=[operator])))
    
    
    #true+=float(np.real(wv_sim.expectation(prep_prog=ansatz(params), pauli_terms=[operator])))
    
    values, exp_val, exp_val_without=mitigator.expectation_value(ansatz(params), update_probabilities=True)
    
    estimate+=exp_val
    not_mitigated+=exp_val_without
    

estimate/=N
not_mitigated/=N
true/=N
error=np.abs(estimate-true)


plt.plot(error, label='absolute error')

plt.xlabel('# steps')
plt.ylabel('Error of estimated expectation value')
plt.yscale('log')
plt.xscale('log', base=2)
plt.xlim(left=2**5)
plt.legend()
plt.savefig("plot_mitigator.png", dpi=150)
plt.show()

#%%

operator=sZ(0)*sZ(1)
#probabilities={}
probabilities={0:(0.9,0.8)}
mitigator=TwoQubitMitigator(operator, device=QPU, noisy_readout_probabilities=probabilities)



p=Program()
p+=X(1)
p+=H(0)
p+=CNOT(0,1)

test_dict=mitigator.update_bit_flip_probabilities(False)
print(test_dict)
#%%
qpu_name = '8q-qvm' # 8q-qvm   Aspen-8
asqvm = True
isnoisy = False
QPU = get_qc(qpu_name,as_qvm=asqvm, noisy=isnoisy)
program_initialization = Program('PRAGMA INITIAL_REWIRING "NAIVE"')
operator=sZ(0)*sZ(1)
probabilities={}
#probabilities={0:(0.9,0.8)}
mitigator=TwoQubitMitigator(operator, device=QPU, num_shots_evaluation=2**15,noisy_readout_probabilities=probabilities)
ansatz_circuit=ansatz(params)
update_probabilities=True



if not mitigator._probabilities or update_probabilities:
    
    probabilities=mitigator.update_bit_flip_probabilities(True)
    
qubits=mitigator.get_qubits()

p0_first=probabilities[str(qubits[0])+'0']
p1_first=probabilities[str(qubits[0])+'1']
p0_second=probabilities[str(qubits[1])+'0']
p1_second=probabilities[str(qubits[1])+'1']

# compute the gammas according to the paper.
gamma_Z1=1-p0_first-p1_first
gamma_Z2=1-p0_second-p1_second
gamma_I1=p1_first-p0_first
gamma_I2=p1_second-p0_second

# compute the coefficients
coeffs=np.array([1/(gamma_Z1*gamma_Z2),
                 gamma_I1/(gamma_Z1*gamma_Z2),
                 gamma_I2/(gamma_Z1*gamma_Z2),
                 (gamma_I1*gamma_I2)/(gamma_Z1*gamma_Z2)])

# create expectation values and set them to zero
# expectation_values_mitigated is an array containing all 4 terms of the mitigation procedure.
values_mitigated=np.zeros((mitigator._num_shots_evaluation,4))
expectation_value_mitigated=np.zeros((mitigator._num_shots_evaluation,1))
expectation_value_without=np.zeros((mitigator._num_shots_evaluation,1))



terms=mitigator._terms

for term in terms:
    
    # initiate expectation values of Z\otimes Z, Z\otimes I and I\otimes Z
    val=np.array([0,0,0,1], dtype=np.float64)
    
    for n,single_result in enumerate(mitigator._evaluate_single_term(ansatz_circuit, term)):
        # separate results for first and second qubit involved
        r1, r2 = single_result
        # update the expectation value of Z\otimes Z and the other operators
        val[0] += (-1)**(r1+r2)
        val[1] += (-1)**r2
        val[2] += (-1)**r1
    

        values_mitigated[n] += np.real(term.coefficient*np.multiply(coeffs,val)/(n+1))
        expectation_value_mitigated[n] += np.real(term.coefficient*np.dot(coeffs, val)/(n+1))
        expectation_value_without[n] += np.real(term.coefficient*val[0]/(n+1))
    
        
        
    
print("True: ", true, "Est: ",expectation_value_mitigated[-1])