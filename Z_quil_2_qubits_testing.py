import numpy as np
from pyquil import get_qc
from pyquil.quil import Program
from pyquil.gates import RX, X
import matplotlib.pyplot as plt
import tqdm

# Get our QuantumComputer instance, with a Quantum Virtual Machine (QVM) backend
qpu_name = '8q-qvm' # 8q-qvm   Aspen-8
asqvm = True
isnoisy = False
QPU = get_qc(qpu_name,as_qvm=asqvm, noisy=isnoisy)

calibration_trials = 2**13
measurement_trials = 2**13


program_initialization = Program('PRAGMA INITIAL_REWIRING "NAIVE"') #changed syntax according to documentation

"""
defining ansatz
"""
def calibration_ansatz(state,prog_in = program_initialization):
    prog_out = prog_in.copy()
    if state == 1:
        prog_out += X(0)
        prog_out += X(1)
    #prog_out.define_noisy_readout(0, p00=0.9, p11=0.8)
    #prog_out.define_noisy_readout(1, p00=0.95, p11=0.85)
    return prog_out

def ansatz(angle_1,angle_2,prog_in = program_initialization):
    prog_out = prog_in.copy()
    prog_out += RX(angle_1, 0)
    prog_out += RX(angle_2, 1)
   # prog_out += CNOT(0,1)
    #prog_out.define_noisy_readout(0, p00=0.9, p11=0.8)
    #prog_out.define_noisy_readout(1, p00=0.95, p11=0.85)
    return prog_out

# Function to replace the run_and_measure function from pyquil which seems to create problems with Aspen-8
def run_and_measure_wrapped(device, prog_in, trials=calibration_trials):
    result=[]
    p=prog_in.copy()
    ro = p.declare('ro', 'BIT', 2)
    p += Program().measure(0, ro[0]) # Measure qubit #0 a number of times
    p += Program().measure(1, ro[1]) # Measure qubit #1 a number of times
    p.wrap_in_numshots_loop(shots=trials)
    p.define_noisy_readout(0, p00=1, p11=0.9)
   # p.define_noisy_readout(1, p00=0.9, p11=0.9)
    # We see probabilistic results of about half 1's and half 0's
    result=QPU.run(QPU.compile(p))
    return result


"""
Function that computes the true, noise-free expectation value of Z\otimes Z given the two angles.
"""
def true_expectation(angle_1, angle_2):

    return np.cos(angle_1)*np.cos(angle_2)



test_angles=[0, np.pi/2, np.pi]

test_angles_2=np.arange(0, 2*np.pi, 0.2)

#define random angles for rotations on qubits 1 and 2
"""rand_angle_1 = 2*np.pi*np.random.random()
rand_angle_2 = 2*np.pi*np.random.random()"""


for rand_angle_1 in tqdm.tqdm(test_angles):
    

    list_of_errors=[]
    
    for rand_angle_2 in test_angles_2:
        

    
        
        
        """
        outfile = open("Z_"+qpu_name+"_asqvm_"+str(asqvm)+"_"+str(rand_angle_1)+"_"+str(rand_angle_2)+".txt","w")
        outfile.write("True value: "+str(true)+"\r\n")
        outfile.close()
        """
        
        """
        The following is based on eq. (40) in the paper. 
        """
        
        
        
        
        # calibration for 0 state
        cal0 = run_and_measure_wrapped(device=QPU, prog_in=calibration_ansatz(0), trials=calibration_trials).T
        # cal0 is dictionary of qubit measurements
        # cal0[q] is list of outputs of qubit q
        # number of ones in cal0[q] is sum(cal0[q])
        # number of zeros is cal0[q] is len(cal0[q]) - sum(cal0[q])
        # pq0 is number ones / len(cal0[q])
        p00 = float(sum(cal0[0]))/len(cal0[0])
        p10 = float(sum(cal0[1]))/len(cal0[1])
        
        
        # calibration for 1 state
        cal1 = run_and_measure_wrapped(device=QPU, prog_in=calibration_ansatz(1), trials=calibration_trials).T
        # number zeros is len(cal1[q]) - sum(cal1[q])
        p01 = float(len(cal1[0]) - sum(cal1[0]))/len(cal1[0])
        p11 = float(len(cal1[1]) - sum(cal1[1]))/len(cal1[1])
        
        
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
        
        
        
        
        results = run_and_measure_wrapped(device=QPU, prog_in=ansatz(rand_angle_1,rand_angle_2), trials=measurement_trials)
        #combine results of first two qubits for further computation
        
        
        estimate=0. 
        not_mitigated=0.
        

        
        test_angles=[-np.pi, -np.pi/2, -np.pi/4, 0,np.pi/4, np.pi/2, np.pi]
        
        
        for n,result in enumerate(results):
            
            res0,res1=result
            
            rand_idx = np.random.choice(range(4),p=Omega)
            
            M=np.array([[1,1],[0,1],[1,0],[0,0]])
            
            m=M[rand_idx]
            
            # If the corresponding operator contains a Z on qubit q, we want to add the value (-1)^j (multiplied
            # by the corresponding sign and the overall factor of course), 
            # where j is the outcome of the measurement for qubit q. Otherwise, we want to add 1 (eigenvalue of Id)
            # We encode this via a list M with entries of the form [m1,m2], where m1,m2=0,1.
            # The correct factor is obtained by taking (-1)^(j*m), which gives -1 only in the case that the 
            # operator is Z AND the outcome is 1.
            estimate += sigma[rand_idx] * (-1)**(res0*m[0]+res1*m[1])
            

            
            
            # compute result without mitigation
            

        
        # normalize to number of measurements
        
        estimate*=sum_of_abs/float(measurement_trials)
        not_mitigated/=float(measurement_trials)
        
        
        true=true_expectation(rand_angle_1, rand_angle_2)
        list_of_errors.append(estimate-true)
        #+0.1*np.cos(rand_angle_1)-0.1*np.cos(rand_angle_2)
        


    plt.plot(test_angles_2, list_of_errors, label="theta_1="+str(np.round(rand_angle_1, decimals=2)))

    
    
plt.xlabel('theta_2')
plt.ylabel('error')
#plt.axis([0,6.2,-0.2,0.2])
plt.legend()
plt.savefig("plot_p00=0,p01=0.1.png", dpi=150)
plt.show()


