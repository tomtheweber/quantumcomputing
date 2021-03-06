import numpy as np
from pyquil import get_qc
from pyquil.quil import Program
from pyquil.gates import RX, RZ, X, CNOT
import matplotlib.pyplot as plt
from pyquil.paulis import sZ
from pyquil.api import WavefunctionSimulator
import pandas as pd

import tqdm





# Get our QuantumComputer instance, with a Quantum Virtual Machine (QVM) backend
qpu_name = '8q-qvm' # 8q-qvm   Aspen-8
asqvm = True
isnoisy = False
QPU = get_qc(qpu_name,as_qvm=asqvm, noisy=isnoisy)

calibration_trials = 2**13
measurement_trials = 2**13



program_initialization = Program('PRAGMA INITIAL_REWIRING "NAIVE"') #changed syntax according to documentation




def calibration_ansatz(state,prog_in = program_initialization):
    """
    Create quantum circuit for calibration, i.e. measuring bit-flip probabilities
    """
    prog_out = prog_in.copy()
    if state == 1:
        prog_out += X(0)
        prog_out += X(1)
    return prog_out

def ansatz(params,prog_in = program_initialization):
    """
    Create a maximally expressive 2-qubit quantum circuit with minimal amount of parameters (15 rotations)
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


def run_and_measure_wrapped(device, prog_in, trials=calibration_trials):
    
    """
    Function to replace the run_and_measure function from pyquil which seems to create problems with Aspen-8
    """
    result=[]
    p=prog_in.copy()
    ro = p.declare('ro', 'BIT', 2)
    p += Program().measure(0, ro[0])
    p += Program().measure(1, ro[1])
    p.wrap_in_numshots_loop(shots=trials)
    
    
    p.define_noisy_readout(0, p00=0.9, p11=1)
    #p.define_noisy_readout(1, p00=0.8, p11=0.85)    
    result=QPU.run(QPU.compile(p))
    return result



n=10000




estimates=np.zeros(n)
not_mitigated=np.zeros(n)
error=np.zeros(n)
true=np.zeros(n)
overheads=np.zeros(n)

p00=np.zeros(n)
p01=np.zeros(n)
p10=np.zeros(n)
p11=np.zeros(n)


# define the Z\otimes Z operator to be measured
    
zz=[sZ(0)*sZ(1)]
    
# initialise pyquil's simulator for calculation of exact expectation value
wv_sim=WavefunctionSimulator()


for i in tqdm.tqdm(range(n)):


    #define random angles for rotations on qubits 1 and 2
    
    theta_0=2*np.pi*np.random.random()
    theta_1=2*np.pi*np.random.random()
    theta_2=2*np.pi*np.random.random()
    theta_3=2*np.pi*np.random.random()
    theta_4=2*np.pi*np.random.random()
    theta_5=2*np.pi*np.random.random()
    theta_6=2*np.pi*np.random.random()
    theta_7=2*np.pi*np.random.random()
    theta_8=2*np.pi*np.random.random()
    theta_9=2*np.pi*np.random.random()
    theta_10=2*np.pi*np.random.random()
    theta_11=2*np.pi*np.random.random()
    theta_12=2*np.pi*np.random.random()
    theta_13=2*np.pi*np.random.random()
    theta_14=2*np.pi*np.random.random()
    
    
    params=np.array([theta_0,
                     theta_1,
                     theta_2,
                     theta_3,
                     theta_4,
                     theta_5,
                     theta_6,
                     theta_7,
                     theta_8,
                     theta_9,
                     theta_10,
                     theta_11,
                     theta_12,
                     theta_13,
                     theta_14])
    
    
    
    
    
    # compute exact expactation value
    true[i]=float(np.real(wv_sim.expectation(prep_prog=ansatz(params), pauli_terms=zz)))
    
    
    
    # calibration for 0 state
    cal0 = run_and_measure_wrapped(device=QPU, prog_in=calibration_ansatz(0), trials=calibration_trials).T
    # cal0 is dictionary of qubit measurements
    # cal0[q] is list of outputs of qubit q
    # number of ones in cal0[q] is sum(cal0[q])
    # number of zeros is cal0[q] is len(cal0[q]) - sum(cal0[q])
    # pq0 is number ones / len(cal0[q])
    p00[i] = float(sum(cal0[0]))/len(cal0[0])
    p10[i] = float(sum(cal0[1]))/len(cal0[1])
    
    
    # calibration for 1 state
    cal1 = run_and_measure_wrapped(device=QPU, prog_in=calibration_ansatz(1), trials=calibration_trials).T
    # number zeros is len(cal1[q]) - sum(cal1[q])
    p01[i] = float(len(cal1[0]) - sum(cal1[0]))/len(cal1[0])
    p11[i] = float(len(cal1[1]) - sum(cal1[1]))/len(cal1[1])
    
    
    # compute the gammas according to the paper.
    gamma_Z1=1-p00[i]-p01[i]
    gamma_Z2=1-p10[i]-p11[i]
    gamma_I1=p01[i]-p00[i]
    gamma_I2=p11[i]-p10[i]
    
    # compute the coefficients based on eq. (40)    
    # the indices correspond to the following operators:
    # 0 - Z_1 \otimes Z_2
    # 1 - Z_1 \otimes id_2
    # 2 - id_1 \otimes Z_2
    # 3 - id_1 \otimes id_2
    coeffs=[1/(gamma_Z1*gamma_Z2),
            -gamma_I1/(gamma_Z1*gamma_Z2),
            -gamma_I2/(gamma_Z1*gamma_Z2),
            (gamma_I1*gamma_I2)/(gamma_Z1*gamma_Z2)]
    
    # compute the sum of all absolute values of the coefficients   
    sum_of_abs=np.sum(np.abs(coeffs))#
    
    
    
    
    #create the probability distribution as well as a list of the signs of the coeffients
    Omega=[]
    sigma=[]
    for k in coeffs:
        Omega.append(np.abs(k)/sum_of_abs)
        
        if k != 0:
            sigma.append(k/np.abs(k))
        
        else:
            sigma.append(1.0)
    
    
    
    # actual experiment of the ansatz circuit
    results = run_and_measure_wrapped(device=QPU, prog_in=ansatz(params), trials=measurement_trials)

    

    

    
    
    for n,result in enumerate(results):
        
        
        #split result of first and second qubit
        res0,res1=result
        
        
        # sample from the probability distribution
        rand_idx = np.random.choice(range(4),p=Omega)
        
        
        # numoy array for handling the corresponding actions according to the above sampling
        M=np.array([[1,1],[0,1],[1,0],[0,0]])
        
        
        # choose correct index
        m=M[rand_idx]
        
        # If the corresponding operator contains a Z on qubit q, we want to add the value (-1)^j (multiplied
        # by the corresponding sign and the overall factor of course), 
        # where j is the outcome of the measurement for qubit q. Otherwise, we want to add 1 (eigenvalue of Id)
        # We encode this via a list M with entries of the form [m1,m2], where m1,m2=0,1.
        # The correct factor is obtained by taking (-1)^(j*m), which gives -1 only in the case that the 
        # operator is Z AND the outcome is 1.
        estimates[i] += sum_of_abs * sigma[rand_idx] * (-1)**(res0*m[0]+res1*m[1])
        
        # compute result without mitigation
        
        not_mitigated[i]+=(-1)**(res0+res1)
        
    estimates[i]/=measurement_trials
    not_mitigated[i]/=measurement_trials
    overheads[i]=sum_of_abs
    
error=np.abs(estimates-true)
    
data=np.stack([p00,p01,p10,p11,overheads,estimates,not_mitigated,true,error],axis=1)
    
df=pd.DataFrame(data=data,columns=['p00','p01','p10','p11','overhead','estimate','non_mitigated','true','error'])

df.head()
    
    


    

    


    


    
    

# old relict of writing into a file

"""
outfile = open("Z_"+qpu_name+"_asqvm_"+str(asqvm)+"_"+str(rand_angle_1)+"_"+str(rand_angle_2)+".txt","a")
outfile.write("estimated value: "+str(estimate)+"\r\n")
outfile.write("without mitigation: "+str(not_mitigated)+"\r\n") 
outfile.write("p00: "+str(p00)+"\r\n")
outfile.write("p11: "+str(p11)+"\r\n")

outfile.close()
""" 



"""Create a plot illustrating the convergence of the estimated expectation value to the true one"""

"""
plt.plot(np.sort(np.stack((overheads, errors), axis=1), axis=0).T[0], np.sort(np.stack((overheads, errors), axis=1), axis=0).T[1])
#plt.plot(list_of_estimates, label="Mitigated expectation value")
#plt.plot(list_of_not_mitigated, label="Non-mitigated expectation value")
#plt.hlines(y=true, xmin=0, xmax=measurement_trials, colors='orange', label='True value')
plt.xlabel('Overhead')
plt.ylabel('Error')


plt.legend()
plt.savefig("overhead_error_.png", dpi=150)
plt.show()
"""

