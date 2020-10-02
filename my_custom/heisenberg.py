# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:52:12 2020

@author: Hando
"""
from typing import Optional
import numpy as np
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit import execute


class Heisenberg:
    
    def __init__(self,
                 n_x: int,
                 n_y: int,
                 j_x: float,
                 j_y: float,
                 j_z: float,
                 mu: float,
                 cyclic: Optional[bool] = True
                 ) -> None:
        self.n_x=n_x
        self.n_y=n_y
        self.j_x=j_x
        self.j_y=j_y
        self.j_z=j_z
        self.mu=mu
        self.cyclic=cyclic
        
        
   

    def operator(self):

        #Variables
        
        N_x=self.n_x
        N_y=self.n_y
        J_x=self.j_x
        J_y=self.j_y
        J_z=self.j_z
        mu=self.mu
        cyclic=self.cyclic
        
        
        #number of qubits (=number of lattice sites)
        num_qubits=N_x*N_y
    
        #create empty list of Pauli operators
        pauli_list=[]
    
        #create an array containing the interaction strengths
        J=[J_x,J_y,J_z]
    
        #create a vector containing the [z,x]-entries necessary for Pauli operators
        vector_of_pauli_indices=[[0,1],[1,1],[1,0]]
    
        #Interaction term:
        for direction in range(3):
            
            
            #select corresponding indices for the direction we are considering
            pauli_index=vector_of_pauli_indices[direction]
            
            #create all spin operators except the ones on the boundary
            for i in range(N_x-1):
                for j in range(N_y-1):
    
                    #Reset the index arrays to zeros
                    z=[0]*num_qubits
                    x=[0]*num_qubits
                    
                    #Set indices corresponding to the right Pauli operators to 1
                    z[index_to_qubit(i,j,N_x)]=pauli_index[0]
                    z[index_to_qubit((i+1)%N_x,j,N_x)]=pauli_index[0]
                    x[index_to_qubit(i,j,N_x)]=pauli_index[1]
                    x[index_to_qubit((i+1)%N_x,j,N_x)]=pauli_index[1]
                    
                    #Create Pauli operator
                    pauli=Pauli(z=z,x=x)
                    
                    #Connect operator with weight
                    pauli_term=[J[direction],pauli]
                    
                    #Add weighted Pauli operator to the list
                    pauli_list.append(pauli_term)
        
                    #Repeat everything for the other neighbour of the lattice site (i,j)
                    z=[0]*num_qubits
                    x=[0]*num_qubits
                    z[index_to_qubit(i,j,N_x)]=pauli_index[0]
                    z[index_to_qubit(i,(j+1)%N_y,N_x)]=pauli_index[0]
                    x[index_to_qubit(i,j,N_x)]=pauli_index[1]
                    x[index_to_qubit(i,(j+1)%N_y,N_x)]=pauli_index[1]
                    pauli=Pauli(z=z,x=x)
                    pauli_term=[J[direction],pauli]
                    pauli_list.append(pauli_term)
                
                
            #Interaction term on right vertical boundary
            for i in range(N_x-1):
                
                
                    #Reset the index arrays to zeros
                    z=[0]*num_qubits
                    x=[0]*num_qubits
                    
                    #Set indices corresponding to the right Pauli operators to 1
                    z[index_to_qubit(i,N_y-1,N_x)]=pauli_index[0]
                    z[index_to_qubit(i+1,N_y-1,N_x)]=pauli_index[0]
                    x[index_to_qubit(i,N_y-1,N_x)]=pauli_index[1]
                    x[index_to_qubit(i+1,N_y-1,N_x)]=pauli_index[1]
                    
                    #Create Pauli operator
                    pauli=Pauli(z=z,x=x)
                    
                    #Connect operator with weight
                    pauli_term=[J[direction],pauli]
                    
                    #Add weighted Pauli operator to the list
                    pauli_list.append(pauli_term)
                    
            #Interaction term on lower horizontal boundary
            for j in range(N_y-1):
                
                
                    #Reset the index arrays to zeros
                    z=[0]*num_qubits
                    x=[0]*num_qubits
                    
                    #Set indices corresponding to the right Pauli operators to 1
                    z[index_to_qubit(N_x-1,j,N_x)]=pauli_index[0]
                    z[index_to_qubit(N_x-1,j+1,N_x)]=pauli_index[0]
                    x[index_to_qubit(N_x-1,j,N_x)]=pauli_index[1]
                    x[index_to_qubit(N_x-1,j+1,N_x)]=pauli_index[1]
                    
                    #Create Pauli operator
                    pauli=Pauli(z=z,x=x)
                    
                    #Connect operator with weight
                    pauli_term=[J[direction],pauli]
                    
                    #Add weighted Pauli operator to the list
                    pauli_list.append(pauli_term)
    
        	#For cyclic boundary conditions, add corresponding terms
            if cyclic==True:
                if N_x>2:
                    
                    for j in range(N_y):
                        #Reset the index arrays to zeros
                        z=[0]*num_qubits
                        x=[0]*num_qubits
                        
                        #Set indices corresponding to the right Pauli operators to 1
                        z[index_to_qubit(0,j,N_x)]=pauli_index[0]
                        z[index_to_qubit(N_x-1,j,N_x)]=pauli_index[0]
                        x[index_to_qubit(0,j,N_x)]=pauli_index[1]
                        x[index_to_qubit(N_x-1,j,N_x)]=pauli_index[1]
                        
                        #Create Pauli operator
                        pauli=Pauli(z=z,x=x)
                        
                        #Connect operator with weight
                        pauli_term=[J[direction],pauli]
                        
                        #Add weighted Pauli operator to the list
                        pauli_list.append(pauli_term) 
                
                if N_y>2:
                    for i in range(N_x):
                        #Reset the index arrays to zeros
                        z=[0]*num_qubits
                        x=[0]*num_qubits
                        
                        #Set indices corresponding to the right Pauli operators to 1
                        z[index_to_qubit(i,0,N_x)]=pauli_index[0]
                        z[index_to_qubit(i,N_y-1,N_x)]=pauli_index[0]
                        x[index_to_qubit(i,0,N_x)]=pauli_index[1]
                        x[index_to_qubit(i,N_y-1,N_x)]=pauli_index[1]
                        
                        #Create Pauli operator
                        pauli=Pauli(z=z,x=x)
                        
                        #Connect operator with weight
                        pauli_term=[J[direction],pauli]
                        
                        #Add weighted Pauli operator to the list
                        pauli_list.append(pauli_term) 
                
                
        #External field term    
        for j in range(num_qubits):
            #Create Pauli operatoe of z's only acting on corresponding qubits - single term (ext. field)
            z=[0]*num_qubits
            x=[0]*num_qubits
            z[j]=1
            pauli=Pauli(z=z,x=x)
            pauli_term=[mu,pauli]
            pauli_list.append(pauli_term)
        
        operator=WeightedPauliOperator(pauli_list)
        
        
        return operator
    
    
    def evaluation(self,
             wave_function,
             backend,
             statevector_mode=False,
             parameters=[],
             shots=1024,
             wrapping_gates=[]
             ):
        operator=self.operator()
        
        if not wrapping_gates:
        
            evaluation_circuits=operator.construct_evaluation_circuit(wave_function=wave_function,statevector_mode=statevector_mode,circuit_name_prefix='eval')
            evaluation_result=backend.execute(evaluation_circuits)
        #evaluation_result=execute(evaluation_circuits,backend=backend,shots=num_shots).result()
            cost=operator.evaluate_with_result(evaluation_result,statevector_mode=statevector_mode,circuit_name_prefix='eval')

            return np.real(cost[0])
        
        else:
            return "no result"
        
        
    def vqe(self):
        return 0

        
    
def index_to_qubit(i,j,n_x):
        qubit=i+j*n_x
        return qubit
        