import numpy as np
from pyquil import get_qc
from pyquil.quil import Program
from pyquil.gates import RX, X
import tqdm

# Get our QuantumComputer instance, with a Quantum Virtual Machine (QVM) backend
qpu_name = "8q-qvm" # 8q-qvm   Aspen-8
asqvm = True
QPU = get_qc(qpu_name,as_qvm=asqvm)

calibration_trials = 2**13
measurement_trials = 2**13
batches = 2**2

program_initialization = Program('PRAGMA INITIAL_REWIRING "NAIVE"') #changed syntax according to documentation

"""
defining ansatz
"""
def calibration_ansatz(state,prog_in = program_initialization):
    prog_out = prog_in.copy()
    if state == 1:
        prog_out += X(0)
        prog_out += X(1)
        
    # Add readout error. Note that pyquil asks for probabilities of success, not vice versa.    
    prog_out.define_noisy_readout(0, p00=0.9, p11=0.8)
    prog_out.define_noisy_readout(1, p00=0.95, p11=0.85)
    return prog_out

def ansatz(angle_1,angle_2,prog_in = program_initialization):
    prog_out = prog_in.copy()
    prog_out += RX(angle_1, 0)
    prog_out += RX(angle_2, 1)
    
    # Add the same values for readout error
    prog_out.define_noisy_readout(0, p00=0.9, p11=0.8)
    prog_out.define_noisy_readout(1, p00=0.95, p11=0.85)
    return prog_out

# Define random angles for rotations on qubits 1 and 2
rand_angle_1 = 2*np.pi*np.random.random()
rand_angle_2 = 2*np.pi*np.random.random()

# Compute independent factors of true expectation value (just because of line space in program)
true_1 = np.cos(rand_angle_1/2.)**2 - np.sin(rand_angle_1/2.)**2
true_2 = np.cos(rand_angle_2/2.)**2 - np.sin(rand_angle_2/2.)**2

# multiply factors to get noise-free expectation value to get
# (cos^2(theta1/2)-sin^2(theta1/2))*(cos^2(theta2/2)-sin^2(theta2/2))
true=true_1*true_2

outfile = open("Z_"+qpu_name+"_asqvm_"+str(asqvm)+"_"+str(rand_angle_1)+"_"+str(rand_angle_2)+".txt","w")
outfile.write("True value: "+str(true)+"\r\n")
outfile.close()


"""
The following is based on eq. (40) in the paper. 
"""

list_of_results=[]

for batch in tqdm.tqdm(range(batches)):

    # calibration for 0 state
    cal0 = QPU.run_and_measure(calibration_ansatz(0), trials=calibration_trials)
    # cal0 is dictionary of qubit measurements
    # cal0[q] is list of outputs of qubit q
    # number of ones in cal0[q] is sum(cal0[q])
    # number of zeros is cal0[q] is len(cal0[q]) - sum(cal0[q])
    # pq0 is number ones / len(cal0[q])
    p00 = float(sum(cal0[0]))/len(cal0[0])
    p10 = float(sum(cal0[1]))/len(cal0[1])


    # calibration for 1 state
    cal1 = QPU.run_and_measure(calibration_ansatz(1), trials=calibration_trials)
    # number zeros is len(cal1[q]) - sum(cal1[q])
    p01 = float(len(cal1[0]) - sum(cal1[0]))/len(cal1[0])
    p11 = float(len(cal1[1]) - sum(cal1[0]))/len(cal1[1])

    
    # compute the gammas according to the paper.
    gamma_Z1=1-p00-p01
    gamma_Z2=1-p10-p11
    gamma_I1=p01-p00
    gamma_I2=p11-p10
    
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
    sum_of_abs=np.sum(np.abs(coeffs))

    #create the probability distribution as well as a list of the signs of the coeffients
    Omega=[]
    sigma=[]
    for k in coeffs:
        Omega.append(np.abs(k)/sum_of_abs)
        
        if k != 0:
            sigma.append(k/np.abs(k))
        
        else:
            sigma.append(1.0)



    # Run the ansatz circuit
    results_all = QPU.run_and_measure(ansatz(rand_angle_1,rand_angle_2), trials=measurement_trials)
    
    #combine results of first two qubits for further computation
    results=list(zip(results_all[0],results_all[1]))

    # initiate the (non-)mitigated expectation values
    estimate=0. 
    not_mitigated=0.
    
    # pick readout of qubit 1, qubit 2 for each measurement
    for res0,res1 in results:
        
        # choose random number according to probability distribution
        rand_idx = np.random.choice(range(4),p=Omega)
        
        # this is for encoding the way to handle results for each possible outcome of the above sampling
        M=np.array([[1,1],[1,0],[0,1],[0,0]])
        
        m=M[rand_idx]
        
        # If the corresponding operator contains a Z on qubit q, we want to add the value (-1)^j (multiplied
        # by the corresponding sign and the overall factor of course), 
        # where j is the outcome of the measurement for qubit q. Otherwise, we want to add 1 (eigenvalue of Id)
        # We encode this via a list M with entries of the form [m1,m2], where m1,m2=0,1.
        # The correct factor is obtained by taking (-1)^(j*m), which gives -1 only in the case that the 
        # operator is Z AND the outcome is 1.
        estimate += sum_of_abs * sigma[rand_idx] * (-1)**(res0*m[0]+res1*m[1])
        
        # compute result without mitigation
        not_mitigated+=(-1)**(res0+res1)
    
    # normalize to number of measurements
    estimate/=float(measurement_trials)
    not_mitigated/=float(measurement_trials)
        
        
    list_of_results.append(estimate)

    
    outfile = open("Z_"+qpu_name+"_asqvm_"+str(asqvm)+"_"+str(rand_angle_1)+"_"+str(rand_angle_2)+".txt","a")
    outfile.write("estimated value: "+str(estimate)+"\r\n")
    outfile.write("without mitigation: "+str(not_mitigated)+"\r\n") 
    outfile.write("p00: "+str(p00)+"\r\n")
    outfile.write("p11: "+str(p11)+"\r\n")

    outfile.close()
    
    
mean=np.sum(list_of_results)/len(list_of_results)

print("\n Mean:", mean, "\n True:", true)

