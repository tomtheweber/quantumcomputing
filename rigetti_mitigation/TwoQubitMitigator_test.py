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
import time
import numpy as np

qpu_name = '8q-qvm' # 8q-qvm   Aspen-8
asqvm = True
isnoisy = False
QPU = get_qc(qpu_name,as_qvm=asqvm, noisy=isnoisy)


#%%

operator=sZ(0)*sZ(1)
#probabilities={}
probabilities={0:(0.9,0.8)}
mitigator=TwoQubitMitigator(operator, device=QPU, noisy_readout_probabilities=probabilities)



p=Program()
p+=X(1)
p+=H(0)
p+=CNOT(0,1)

val=mitigator.expectation_value(p)

print(val)
