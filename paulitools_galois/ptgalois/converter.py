from ptgalois.core import *
#from paulitools.core import toPauli
#from paulitools.util import multiKron
import numpy as np
import galois
from galois import GF
INTTYPE = int
from numpy import kron
import copy



I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
#Conversion Dictionaries
toMatrix = {0: I, 1: X, 2: Y, 3: Z, 'I': I, 'X': X, 'Y': Y, 'Z': Z}
toPauli = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z', 'I': 0, 'X': 1, 'Y': 2, 'Z': 3}


def toZX(input_list, qubits=None, signs=False):
    """
    Converts Pauli strings into binary symplectic form (ZX)
    
    Args:
        p_list: List of Pauli Matricies, in the following formats:
            Example: $+I_{0}X_{1}Y_{2}Z_{3}$
            1. List of Pauli Integers: [0,1,2,3] = IXYZ
            2. List of Pauli Matrix Letters: ['I', 'X', 'Y', 'Z'] = IXYZ
            3. String of Pauli Matricies: 'IXYZ'
            4. List of Pauli Matrix Letters with qubit index: [('I', 0), ('X', 1), ('Y', 2), ('Z', 3)] = IXYZ
            5. List of Pauli Integers with qubit index: [(0, 0), (1, 1), (2, 2), (3, 3)] = IXYZ
            #Qiskit notation: Lowests qubit is left term in string
            6. A String of binary numbers: Partitioned as $Z_{2,1,0}|X_{2,1,0}$: '11000110' = IXYZ
            other:
            If already using galois, does nothing
            TODO:
            OpenFermion, Qiskit
            List of Pauli Matricies: [I, X, Y, Z] = IXYZ
        qubits: Number of qubits (Required for formats 4 and 5)
        signed (bool or arraylike): If not False, must be a list of the same length as p_list, and will be used to determine the sign of the pauli string. Each entry should be True/False or 1/0 or -/+ (True for -, False for +)
    Returns:
        Array: (2n x len(p_list)) Matrix of Pauli Matricies in the Z and X basis
        Individual pauli strings formatted as: [I_0 X_1 Y_2 Z_3] -> (0011|0110) -> int('11000110',2) = 198
    Throws:
        Exception: If n is not specified, but is needed to compute the matrix
    """
    p_list = copy.deepcopy(input_list)
    #sanitize input
    #Check if the input has type galois:
    GF2 = GF(2)
    format6 = False

    if isinstance(p_list, galois.GF2):
        #Print a warning: Already formatted, returning input:
        #print('Input is already in GF2 format, returning input')
        return p_list
    if isinstance(p_list, np.ndarray):
        return GF2(p_list)
        
    
    #Checks if signs are listed along with the pauli strings. If so, signs will be an list with equal length. Else, it will be the bool "False"
    #Converts bools to ints, keeps ints as ints, and converts strings to ints
    if not signs is False:
        signed = True
        assert(len(signs) == len(p_list)), 'Signed must be the same length as p_list. Nd arrays not yet supported'
        if isinstance(signs[0], bool):
            signs = np.array(signs, dtype=int)
        elif isinstance(signs[0], str):
            sign_to_int = {'+':0, '-':1, '0':0, '1':1}
            signs = np.array([sign_to_int[s] for s in signs], dtype=int)
        #Converts Signs to a GF array:
        signs = GF2(signs)
    else:
        signed = False
    
    # Unifies formatting for single pauli string input by forcing it to be a list
    if isinstance(p_list, np.ndarray):
        #Convert to a list:
        temp = p_list.copy()
        p_list = None
        p_list = temp.tolist()#p_list.tolist()

    if not isinstance(p_list, list):
        p_list = [p_list]

    L = len(p_list) #The number of Pauli strings in the set

    # Reformats 3 -> 2:
    #Splits the string into a list of characters
    if isinstance(p_list[0], str):
        try:
            int(p_list[0])
            format6 = True
        except:
            temp = p_list.copy()
            for i in range(L):
                temp[i] = list(p_list[i])
            p_list = temp
            #temp = None
    #Ensure all the lists are the same length. Prune any which are not:
    p_list = [p for p in p_list if len(p) == len(p_list[0])]
    L = len(p_list)
    #if already format #1, converts to numpy array
    #Converts a list of lists of numbers into a array of ints
    if isinstance(p_list[0][0], int):
        p_list = np.array(p_list, dtype=int)

    
    # Reformats 4 -> 1 (array):
    # Reformats 5 -> 1 (array):
    # Formats both Tuple formats into arrays of the correct size (checking ints and strings
    #Both remaining formats will be arrays
    # If the list is in Tuple format, we need to get n, and also populate a list in the format of another style
    try:
        if isinstance(p_list[0][0], tuple):
            print('Tuple format detected')
            if qubits is None:
                #TODO: find the largest index specified and set that as the length
                Exception('n is not specified, but is needed to compute the matrix')
            else:
                if isinstance(p_list[0][0][0], int):
                    p_list_new = np.zeros((L,qubits), int)
                    for i in range(L): 
                        for tup in p_list[i]:
                            p_list_new[i,tup[1]] = tup[0]
                    
                else:
                    p_list_new = np.zeros((L,qubits), dtype=int)
                    for i in range(L):
                        for tup in p_list[i]:
                            p_list_new[i,tup[1]] = toPauli[tup[0]]
                p_list = p_list_new
    except: #Ignore if failed because it's in a valid format
        pass
    
    #Only formats 4 and 5 need to be told the number of qubits, so we can now directly set it. 
    if qubits is None:
        qubits = len(p_list[0])
        if format6:
            qubits = qubits  // 2

    # Reformats 2 -> 1 (array):
    #Converts strings to integers, and populates a numpy array
    #Need to check that we are not format 6 yet
    if isinstance(p_list[0][0], str):
        if not format6:
            if not isinstance(p_list, np.ndarray):
                p_list = np.asarray(p_list)
            p_list_new = np.zeros((L,qubits), dtype=int)
            for i in range(L):
                for j in range(len(p_list[i])):
                    try:
                        p_list_new[i,j] = toPauli[p_list[i,j]]
                    except Exception as e:
                        print(e)
                        print('Error in converting string to Pauli')
                        print('String:', p_list)
                        print('Input:', input_list)
                        raise e
                        
            p_list = p_list_new
    #print(p_list)
    
    #The only forms remaining are #1 and #6
    
    
    ZX_array = GF2.Zeros((L,2*qubits))
    
    #We start by converting to a string of 0s and 1s:
    #Do X and Z separately, then combine them
    if not format6:
        for i in range(L):
            for j in range(qubits):
                if p_list[i,j] == 2 or p_list[i,j] == 3:
                    ZX_array[i,j] = 1
                    #val = val ^ (1 << j) #XOR: 
                if p_list[i,j] == 1 or p_list[i,j] == 2:
                    ZX_array[i,j+qubits] = 1
                    #val = val ^ (1 << j+qubits) #OR symbol: 
    else:
        for i in range(L):
            for j in range(2*qubits):
                try:
                    ZX_array[i,j] = int(p_list[i][j]) 
                except Exception as e:
                    print(e)
                    print('Error in converting string to Pauli')
                    print('String:', p_list)
                    print('Input:', input_list)
                    print('ZX_array:', ZX_array)
                    raise e
                
    if signed:
        ZX_array = np.concatenate((ZX_array,signs[:,np.newaxis]), axis=1)
    return ZX_array

def toString(paulis):
    """
    Converts a list of ZX Matricies to a list of Pauli Matricies
    Args:
        bsfP: List of ZX Matricies in the form of an integer
        n: Number of qubits
    Returns:
        List: List of Pauli Matricies in the form "XIZY"
    """
    #print(Warning('toString does not yet accommodate signed pauli strings'))
    if isinstance(paulis, list):
        paulis = np.asarray(paulis)
    if len(paulis.shape) == 1:
        paulis = np.array([paulis])
    if paulis.shape[-1] % 2:
        #Strip the 0th index of axis 1 (the sign)
        paulis_return = paulis.copy()
        paulis_return = paulis_return[:,:-1]
    else:
        paulis_return = paulis.copy()
    BSFTOPAULI = np.asarray([['I', 'X'], ['Z', 'Y']])
    p_list = []
    length = paulis_return.shape[0]
    qubits = paulis_return.shape[-1]//2

    for i in range(length):
        char = ''
        for n in range(qubits):
            #Get the nth binary element and the n + qubits binary element:
            paulibsf = paulis_return[i]
            Z = paulibsf[n]
            X = paulibsf[n + qubits]
            #Z = (paulibsf >> n) & 1
            #X = (paulibsf >> (n + qubits)) & 1
            char += BSFTOPAULI[Z,X]
        p_list.append(char)
    return p_list

def ZXtoMatrix(p_list, signed=False, weights = None, return_sum = False):
    """Returns the Matrix form of the Pauli string"""
    assert signed == False, 'Signed pauli strings not yet supported -- Add the signs as weights instead'
    #If there is only one element, make it iterable
    if len(p_list.shape) == 1:
        p_list = [p_list]
    p_str_list = toString(p_list)
    qubits = len(p_str_list[0])
    if weights is None:
        weights = np.ones(len(p_str_list))
    assert len(weights) == len(p_list), 'Weights must be the same length as p_list'
    
    

    if return_sum:
        return_array = np.zeros((2**qubits,2**qubits), dtype=complex)
    else:
        return_array = np.zeros((len(p_list),2**qubits,2**qubits), dtype=complex)
    for i in range(len(p_str_list)):
        p_str = p_str_list[i]
        p_matrix_list = [toMatrix[p_str[j]] for j in range(qubits)]
        p_matrix = multiKron(p_matrix_list)
        if return_sum:
            return_array += weights[i] * p_matrix
        else:
            return_array[i] = p_matrix * weights[i]
    return return_array

        
        
    
    
    
    
    
    
def ZXtoPauli_depreciated(bsfP, qubits):
    """
    Converts a list of ZX Matricies to a list of Pauli Matricies
    Args:
        bsfP: List of ZX Matricies in the form of an integer
        n: Number of qubits
    Returns:
        List: List of Pauli Matricies in the form "XIZY"
    """
    BSFTOPAULI = np.asarray([['I', 'X'], ['Z', 'Y']])
    p_list = []
    if type(bsfP) == int:
        bsfP = [bsfP]
    try:
        length = bsfP.shape[0]
    except:
        try: 
            length = len(bsfP)
        except:
            length = bsfP.length
        

        
    for i in range(length):
        char = ''
        for n in range(qubits):
            #Get the nth binary element and the n + qubits binary element:
            paulibsf = bsfP[i]
            Z = (paulibsf >> n) & 1
            X = (paulibsf >> (n + qubits)) & 1
            char += BSFTOPAULI[Z,X]
        p_list.append(char)
    return p_list   

# Converter Functions
def multiKron(P):
    """ Applies the Kronecker Product to a list of Pauli Matrices

    Parameters
    ----------
    P : list
        A list of Pauli Matrices

    Returns
    -------
    K : numpy array
        The Kronecker Product of the Pauli Matrices
    """
    K = P[0]
    for i in range(1,len(P)):
        K = kron(K, P[i])
    return K
