import numpy as np
from pyquil import get_qc
from pyquil.quil import Program, Pragma
from pyquil.gates import RX, RZ, X
import tqdm

# Get our QuantumComputer instance, with a Quantum Virutal Machine (QVM) backend
qpu_name = "8q-qvm" # 8q-qvm   Aspen-8
asqvm = True
QPU = get_qc(qpu_name,as_qvm=asqvm)

calibration_trials = 2**10
measurement_trials = 2**10
batches = 2**2

program_initialization = Program(Pragma('INITIAL_REWIRING', ['"NAIVE"']))

"""
defining ansatz
"""
def calibration_ansatz(state,prog_in = program_initialization):
    prog_out = prog_in.copy()
    if state == 1:
        prog_out += X(0)
    return prog_out

def ansatz(angle,prog_in = program_initialization):
    prog_out = prog_in.copy()
    prog_out += RX(angle,0)
    return prog_out

rand_angle = 2*np.pi*np.random.random()
# state = cos(angle) |0> - i sin (angle) |1>
# <state| Z |state> = cos^2(angle/2.) - sin^2(angle/2.)
true = np.cos(rand_angle/2.)**2 - np.sin(rand_angle/2.)**2
outfile = open("Z_"+qpu_name+"_asqvm_"+str(asqvm)+"_"+str(rand_angle)+".txt","w")
outfile.write(str(true)+"\r\n")
outfile.close()

for batch in tqdm.tqdm(range(batches)):
    p_list = []
    t0 = []
    t1 = []
    trand = []
    cal0 = QPU.run_and_measure(calibration_ansatz(0), trials=calibration_trials)
    # cal0 is dictionary of qubit measurements
    # cal0[q] is list of outputs of qubit q
    # number of ones in cal0[q] is sum(cal0[q])
    # number of zeros is cal0[q] is len(cal0[q]) - sum(cal0[q])
    # p0 is number ones / len(cal0[q])
    p0 = float(sum(cal0[0]))/len(cal0[0])
    p_list.append(p0)

    cal1 = QPU.run_and_measure(calibration_ansatz(1), trials=calibration_trials)
    # number zeros is len(cal1[q]) - sum(cal1[q])
    p1 = float(len(cal1[0]) - sum(cal1[0]))/len(cal1[0])
    p_list.append(p1)

    # E Z^n = (1-p0-p1) Z + (p1-p0) Id
    # Z = 1/(1-p0-p1) Z^n + (p0-p1)/(1-p0-p1) Id

    # T0 = 1./(1.-p0-p1) * sZ(0)
    # T1 = T0 + (p0-p1)/(1.-p0-p1) * ID

    omega = [ 1./(1.-p0-p1) , (p0-p1)/(1.-p0-p1) ]
    OMEGA = abs(omega[0]) + abs(omega[1])
    Omega = []
    sigma = []
    for o in omega:
        Omega.append(abs(o)/OMEGA)
        if o != 0.:
            sigma.append(o/abs(o))
        else:
            sigma.append(1.)

    # ops = [ OMEGA*sigma[0]*sZ(0) , OMEGA*sigma[1]*ID ]

    results = QPU.run_and_measure(ansatz(rand_angle), trials=measurement_trials)

    new_t0 = 0.
    new_trand = 0.
    for res in results[0]:
        new_t0 += omega[0] * ((-1)**res)
        rand_idx = np.random.choice(range(2),p=Omega)
        new_trand += OMEGA * sigma[rand_idx] * ( (1-rand_idx) * ((-1)**res) + rand_idx )
        
    t0.append( new_t0 / float(measurement_trials) )
    t1.append( t0[-1] + omega[1] )
    trand.append( new_trand / float(measurement_trials) )
    
    outfile = open("Z_"+qpu_name+"_asqvm_"+str(asqvm)+"_"+str(rand_angle)+".txt","a")
    outfile.write(str(p_list)[1:-1]+"\r\n")
    outfile.write(str(t0)[1:-1]+"\r\n")
    outfile.write(str(t1)[1:-1]+"\r\n")
    outfile.write(str(trand)[1:-1]+"\r\n")
    outfile.close()
