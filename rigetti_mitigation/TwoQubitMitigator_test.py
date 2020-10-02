# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:48:11 2020

@author: Hando
"""

from pyquil.paulis import sZ, sX, sY, PauliSum, PauliTerm, commuting_sets
from rigetti_mitigation.TwoQubitMitigator import TwoQubitMitigator
from pyquil import get_qc
from pyquil.quil import Program
from pyquil.gates import RX, RZ, X,Y,Z, CNOT
import time
import numpy as np

qpu_name = '8q-qvm' # 8q-qvm   Aspen-8
asqvm = True
isnoisy = False
QPU = get_qc(qpu_name,as_qvm=asqvm, noisy=isnoisy)


#%%

operator=sZ(0)*sZ(1)
single_term=2*sX(0)*sY(1)
probabilities={0:(0.9,0.2)}
mitigator=TwoQubitMitigator(operator, device=QPU, noisy_readout_probabilities=probabilities)



p=Program()
p+=X(1)
p+=X(5)


tic=time.time()
probs=mitigator.get_bit_flip_probabilities(True)

toc=time.time()

print('Time needed for program: ',toc-tic, ' seconds.' )

print(mitigator._evaluate_single_term(p, single_term)[0])

#%%
for term in operator.terms:
    print(term.coefficient)

