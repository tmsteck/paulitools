from ptgalois.converter  import *
from ptgalois.core import inner_product, commutator
from ptgalois.util import powers_of_two_between, popcount, pruneGroup
#from numba import jit

class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.v = val
        
class Tree:
    def __init__(self):
        self.root = None
        self.length = 0
        self.k = None
    
    def __len__(self):
        return self.length

    def getRoot(self):
        return self.root
    
    def add(self, val):
        if self.root is None:
            self.root = Node(val)
            self.length += 1
            return 1
        else:
            return self._add(val, self.root)

    def _add(self, val, node):
        if val < node.v:
            if node.l is not None:
                return self._add(val, node.l)
            else:
                node.l = Node(val)
                self.length += 1
                return 1
        else:
            if val == node.v:
                return 0
            elif node.r is not None:
                return self._add(val, node.r)
            else:
                node.r = Node(val)
                self.length += 1
                return 1
    
    def find(self, val):
        if not self.root is None:
            return self._find(val, self.root)
        else:
            return None

    def _find(self, val, node):
        if val == node.v:
            return node
        elif (val < node.v and node.l is not None):
            return self._find(val, node.l)
        elif (val > node.v and node.r is not None):
            return self._find(val, node.r)

    def inTree(self, val):
        if not self.root is None:
            return self._inTree(val, self.root)
        else:
            return 0

    def _inTree(self, val, node):
        if val == node.v:
            return 1
        elif (val < node.v and node.l is not None):
            return self._inTree(val, node.l)
        elif (val > node.v and node.r is not None):
            return self._inTree(val, node.r)
        else:
            return 0
    

    def deleteTree(self):
        # garbage collector will do this for us. 
        self.root = None
    
    def __iter__(self):
        self.nodes = []
        self._inorder_traversal(self.root)
        return iter(self.nodes)
    
    def _inorder_traversal(self, node):
        if node is not None:
            self._inorder_traversal(node.l)
            self.nodes.append(node)
            self._inorder_traversal(node.r)
    
    def copy(self):
        #Return a deep copy of the binary tree:
        newTree = Tree()
        newTree.root = self._copy(self.root)
        newTree.length = self.length
        return newTree
    
    def _copy(self, node):
        if node is None:
            return None
        else:
            newNode = Node(node.v)
            newNode.l = self._copy(node.l)
            newNode.r = self._copy(node.r)
            return newNode
    
def generator(paulis):
    """ Finds a set of Paulis which generate the input set or group. Generates under multiplication or equivalently BSF addition (mod 2)
    
    Performs Row Reduction on the Paulis to find a set of generators.
    Args:
        Paulis: BSF of the Paulis using the standard Galois matrix representation
    Returns:
        Galois Matrix of the generators
    """
    return paulis.row_reduce().row_space()
    

def findGenerators_depreciated(paulis, n):
    """ Finds a set of Paulis which generate the input set or group. Generates under multiplication or equivalently BSF addition (mod 2)
    
    Performs Row Reduction on the Paulis to find a set of generators.
    Args:
        Paulis: Binary Symplectic Form of Paulis formatted as entries in a 1D numpy array
        n (int): Number of qubits
    Returns:
        ndarray (int): Binary Symplectic Form of the generators
    """
    #Step 1: Sort the paulis so larger integers are closer to index 0
    paulis = np.sort(paulis)[::-1]
    #Initializes the While loop: Waits until the largest devisor is less than 1:
    #Finds the largest power of two divisor
    binary_index = int(np.ceil(np.log2(paulis[0]+1)))-1 #Bit_length but for numpy ints
    index = 0
    while binary_index > -1:
        #Step 2: Check through each pauli: If it has a 1 in the binary index, that add the first pauli to it
        for i in range(index+1, len(paulis)): #Ignores the first paulis we don't need to row reduce
            #print(paulis[i] >> binary_index, paulis[i])
            if paulis[i] >> binary_index: #Checks the value of the digit at the binary index -- anything larger than 2^binary_index will have already been dealt with
                paulis[i] = paulis[i] ^ paulis[index]
        paulis = np.sort(paulis)[::-1]
        #print(paulis)
        #print(ZXtoPauli(paulis, n))
        #Step 3: Find the next largest power of two divisor
        index += 1
        if paulis[index] == 0:
            #print(paulis[index])
            break
        binary_index = int(np.ceil(np.log2(paulis[index]+1)))-1
        
    #Step 4: Remove all zeros from the array:
    paulis_new = np.zeros(index, dtype=INTTYPE)
    paulis_new[:] = paulis[:index]
    print('Finished Row Reduction')
    print(paulis_new)
    print(ZXtoPauli(paulis_new, n))
    return paulis_new

def rowReduce_depreciated(paulis, n, auxilliary=None):
    """
    Computes the Gauss-Jordan form of the paulis in the binary symplectic form. Performs the same operations on the auxiliary matrix
    If auxiliary is none, generates a vector of zeros. Auxilliary in general is a vector of binary elements. This is not stored as an integer but instead a list of integers
    Section 1: Input checking
    1. Checks if Auxilliary is not none
    2. If not None, checks that it is vector with the same dimension as the array of paulis, which is a vector of integers
    3. If not None, checks that the auxilliary comprises integers
    4. If None, generates an array of zero integers
    Section 2:
    1. Sort paulis in decending order, and sort auxilliary according to the same ordering (so values at equal indices remain associated)
    2. Finds the bit length of the largest integer in paulis
    3. For each integer in paulis that has a 1 at the value of bit_length, add it to the counting index
    4. resort the array, continue 1-3 until the largest integer is 0
    Section 3: Back substitution
    1. Starts at the smallest integer in paulis
    2. For each other element in the list which has a 1 at the bit_length of the current integer, add the current integer to it
    3. Repeat for each integer in paulis
    
    
    Args:
        paulis (array): an array of 64 bit integers
        n (int): Number of qubits
        auxilliary (array): an array of integers
    Returns:
        (ndarray, ndarray) (tuple of integer arrays): Binary Symplectic Form of the generators, Binary Symplectic Form of the auxilliary
    """
    #Input checking
    if auxilliary is not None:
        if len(auxilliary) != len(paulis):
            raise ValueError('Auxilliary must be the same size as the paulis')
        if not isinstance(auxilliary, np.ndarray):
            try:
                auxilliary = np.asarray(auxilliary)
            except:
                raise ValueError('Auxilliary must be a numpy array')
        if auxilliary.dtype != INTTYPE:
            raise ValueError('Auxilliary must be an array of integers')
    else:
        auxilliary = np.zeros(len(paulis), dtype=INTTYPE)
        
    sort_indices = np.argsort(paulis)[::-1]
    paulis = paulis[sort_indices]
    auxilliary = auxilliary[sort_indices]
    
    binary_index = int(np.ceil(np.log2(paulis[0]+1)))-1 #Bit_length but for numpy ints
    index = 0
    while binary_index > -1:
        #Step 2: Check through each pauli: If it has a 1 in the binary index, that add the first pauli to it
        for i in range(index+1, len(paulis)): #Ignores the first paulis we don't need to row reduce
            #print(paulis[i] >> binary_index, paulis[i])
            if paulis[i] >> binary_index: #Checks the value of the digit at the binary index -- anything larger than 2^binary_index will have already been dealt with
                paulis[i] = paulis[i] ^ paulis[index]
                auxilliary[i] = auxilliary[i] ^ auxilliary[index]
        
        sort_indices = np.argsort(paulis)[::-1]
        paulis = paulis[sort_indices]
        auxilliary = auxilliary[sort_indices]
        
        #Step 3: Find the next largest power of two divisor
        index += 1
        if paulis[index] == 0:
            break
        binary_index = int(np.ceil(np.log2(paulis[index]+1)))-1
    #Cuts out the elements which 
    #We now have the row form of the paulis, we still want the reduced row echelon form
    #Now, we need to do back-substitution
    
    sort_indices = np.argsort(paulis)
    paulis = paulis[sort_indices]
    auxilliary = auxilliary[sort_indices]
    
    def check_bit_at_index(n, index):
        mask = 1 << index  # Create a mask with a one at the specified index
        return (n & mask) != 0  # Check if the bit at the index is one

    binary_index = int(np.ceil(np.log2(paulis[0]+1)))-1 #Bit_length but for numpy ints
    index = 0
    while binary_index > -1:
        #Step 2: Check through each pauli: If it has a 1 in the binary index, that add the first pauli to it
        for i in range(index+1, len(paulis)): #Ignores the first paulis we don't need to row reduce
            #print(paulis[i] >> binary_index, paulis[i])
            if check_bit_at_index(paulis[i], binary_index): #Checks the value of the digit at the binary index -- anything larger than 2^binary_index will have already been dealt with
                paulis[i] = paulis[i] ^ paulis[index]
                auxilliary[i] = auxilliary[i] ^ auxilliary[index]
        
        sort_indices = np.argsort(paulis)
        paulis = paulis[sort_indices]
        auxilliary = auxilliary[sort_indices]
        
        #Step 3: Find the next largest power of two divisor
        index += 1
        if paulis[index] == 0:
            break
        binary_index = int(np.ceil(np.log2(paulis[index]+1)))-1
    return paulis, auxilliary
    

def destabilizer_depreciated(paulis, n, reduced=False):
    """
    Computes the decentralizers from a set of generators. Completes the basis to the full set of Paulis
    
    Algorithm:
    (0.) If the input is not in reduced row echelon form, reduce it
    1. Sort the integers in decending order
    2. index i = 0
    3. Evaluate the largest power of two divisor of the integer at index i
    4. index i + 1
    5. Evaluate the largest power of two divisor of the integer at index i
    6. Determine the powers of two in the gap between (see funciton powers_of_two_between)
    7. Add these powers of two to the list
    8. set i = i+1, repeat from step 4 until exhausted
    
    Args:
        paulis (array): an array of 64 bit integers. Assumes Paulis are generators and linearly independent
        reduced (bool): If True, assumes the input is in reduced row echelon form
    Returns:
        ndarray (int): Binary Symplectic Form of the decentralizers    
    """
    if not reduced:
        paulis, auxilliary = rowReduce(paulis, n)
    #sort the paulis in decending order
    paulis = np.sort(paulis)[::-1]
    #Initializes the While loop: Waits until the largest devisor is less than 1:
    #Finds the largest power of two divisor
    binary_index = int(np.ceil(np.log2(paulis[0]+1)))-1 #Bit_length but for numpy ints
    index = 0
    index_max = len(paulis) 
    destabilizers = []
    next_binary_index = int(np.ceil(np.log2(paulis[index + 1]+1)))-1
    while binary_index > -1:
        destabilizers += powers_of_two_between(1 << (binary_index-1), 1 << (next_binary_index-1))
        binary_index = next_binary_index
        index += 1
        if index > index_max:
            break
        next_binary_index = int(np.ceil(np.log2(paulis[index + 1]+1)))-1
    destabilizers = np.asarray(destabilizers)
    destabilizers = np.unique(destabilizers)
    destabilizers = np.sort(destabilizers)[::-1]
    return destabilizers

def radical(paulis,reduced=False):
    """
    Computes the generators of the radical of the group generated by the paulis
    f
    equal to the interaction of the group with the complement
    
    returns <p>^TS<p> whedre <p> is the tableau of the paulis and S is the symplectic matrix
    """
    if not reduced:
        paulis = paulis.row_reduce().row_space()
    #print('GSG', inner_product(paulis, paulis))
    return inner_product(paulis, paulis).null_space()
    
def centralizer(paulis, reduced=False):
    ker = radical(paulis, reduced=reduced)
    #print('ker',ker)
    if reduced:
        return ker @ paulis
    else:
        return ker @ paulis.row_reduce().row_space()
    
def logical(paulis, reduced=False):
    updated = paulis.row_reduce().row_space()
    row_space = inner_product(updated, updated).row_space()
    return row_space @ paulis

def inGroup(element, generators, reduced_generators=True):
    """Given a single Pauli string and a list of generators, checks if the element is generated by the group. Performs row reduction with the modified group, and checks if it is the same length

    Args:
        element (int): Pauli string to check
        generators (ndarray): List of generators
        reduced_generators (bool): If True, assumes the generators are a minimal set of generators and don't require row-reduction initially
    Returns:
        bool: True if the element is in the group, False otherwise
    """
    
    #Check if element is a string
    if isinstance(element, str):
        element = toZX(element)
    if not reduced_generators:
        generators = generator(generators)
    target_length = len(generators)
    merged_generators = np.concatenate((generators, element))#np.asarray([element])))
    new_length = merged_generators.row_reduce().row_space().shape[0]

    if new_length == target_length:
        return True
    else:
        return False
    #return element in generators




def centralizerLC_depreciated(paulis,n):
    """
    Given a set of generators $\{g_i\}$ of a group $G$, attempts to find the centralizer of the group $C$. 
    Assumes the generator of the group is of the form $\left< C, L \right>$ where $C$ is the set of centralizers and $L$ is the set of 
    so-called ''Logical Operators.'' 
    
    Performs the following:
    1. Performs Gaussian Elimination on the generators
    2. Finds the destabilizer generators which combined generate $\mathcal{P}_n$
        a. From the reduced-row echelon form, 
    Using the property that:
    {c,d} = 0
    [c,c] = [d,d] = 0
    Sets up the matrix:
    [ gz | gx ] [ 0  | I ] [cz] = [0]
    [ dz | dx ] [ -I | 0 ] [cx]   [1]
    the solution to which is given via Gaussian elimination of:
    [ gz | gx ]@[ 0  | I || 0 ] 
    [ dz | dx ] [ -I | 0 || 1 ]
    
    Args:
        paulis: Binary Symplectic Form of Paulis formatted as entries in a 1D numpy array
        n (int): Number of qubits
    Returns:
        ndarray (int): Binary Symplectic Form of the centralizer
    """ 
    paulis, aux = rowReduce(paulis, n)
    destabilizers = destabilizer(paulis, n, reduced=True)
    #Left multiply by the Symplectic matrix:
    assert paulis.shape[0] + destabilizers.shape[0] == 2*n, 'The number of generators ({}) and destabilizers ({}) must sum to 2n'.format(paulis.shape[0], destabilizers.shape[0])
    #TODO construct Symplectic Transposed matrix, use ^ + popcount for the multiplication
    full_matrix = np.concatenate((paulis, destabilizers))
    sym_inner_transpose = np.zeros(2*n, dtype=INTTYPE)
    for i in range(0,2*n):
        sym_inner_transpose[i] = 1<<((n-i-1) % (2*n)) 
    output_matrix = np.zeros(2*n, dtype=INTTYPE)
    for i in range(2*n):
        output = 0
        for j in range(2*n):
            output = output ^ ((popcount(full_matrix[j] ^ sym_inner_transpose[i]) % 2) << i)
        output_matrix[j] = output
    #Apply Gauss Jordan Elimination
    auxilliary = np.array(2**n-1, dtype=INTTYPE)
    reduced = rowReduce(output_matrix, n, auxilliary=auxilliary)
    #TODO: FIND THE KERNEL OF THE REDUCED MATRIX
    return reduced 
    

def generateGroup(paulis, n, steps=-1, debug=False):
    """ 
    Generates a group under multiplication or equivalently BSF addition (mod 2) from a set of Paulis
    Args:
        Paulis: Binary Symplectic Form of Paulis formatted as entries in a 1D numpy array
        n (int): Number of qubits
        steps: Number of steps to generate the group, -1 for infinite
    Returns:
        ndarray (int): Binary Symplectic Form of the generators
    """
    group = set()
    for i in range(len(paulis)):
        #print(toString(paulis[i]))
        group.add(toString(paulis[i])[0])
    ZX_paulis = toZX(list(group))
    old_group_length = -1
    current_group_length = len(group)
    while not old_group_length == current_group_length:
        for i in range(len(ZX_paulis)):
            for j in range(i+1,len(ZX_paulis)):
                p = ZX_paulis[i] + ZX_paulis[j]
                string = toString(p)
                group.add(string[0])
        if debug:
            print('Step Number: ', steps)
            print('Group Length: ', len(group))
            print(old_group_length, current_group_length)
            print(group)
        old_group_length = current_group_length
        current_group_length = len(group)
        if debug:
            print(old_group_length, current_group_length)
        steps -= 1
        if steps == 0:
            break
        ZX_paulis = toZX(list(group))
    output = toZX(list(group))
    #print(output[0])
    return output
    

#@jit(nopython=False, parallel=True)
def generateAlgebra(Paulis, n, steps=-1):
    """
    Generates the group of Pauli Matricies
    #TODO: Fix the output -- currently tailored for Cartan decomposition
    Args:
        Paulis: Pauli Matricies to generate the group from
        n (int): Number of qubits
        steps: Number of steps to generate the group, -1 for infinite
    Returns:
        ndarray (int): Pauli Matricies in the group
    """
    end_flag = 2
    #The first group is just the Hamiltonian
    group1 = Tree()
    #We store each iteration separately to increase efficiency of comparisons
    groups = [group1]
    #The number of iterations keeps track of which order commutator we are doing
    iterationNumber = 1
    #Copies the hamiltonian into the first group
    for p in Paulis:
        group1.add(p)
    
    #Specifies that the first set is in m
    group1.k = 0
        
    #Counting the number of operations to determine the speed:
    CommutatorCounter = 0
    searchCounter = 0
    
    
    #Start building the group:
    #Termination conditions are:
    # 1. We have reached the desired number of steps
    # 2. We have reached a point where no new elements are added
    #The second condition flags when the newgroup has no elements
    
    #trial h's TBD how many we should do:
    #Pick term in the Hamiltonian, check what else it commutes with:
    h_list = []
    for p in group1:
        h = Tree()
        h.add(p.v)
        for q in group1:
            c = 0
            for g in h:
                #If it commutes with one of the elements already in h, break
                c = commutator(g.v, q.v, n)
                if c != 0:
                    break
            #Only add if commutes with everything
            if c == 0:
                h.add(q.v)
        h_list.append(h)
        
    #print([[ZXtoPauli([p.v for p in h], n)] for h in h_list])
    
    while steps != 0:
        for i in range(iterationNumber):
            groupa = groups[i]
            groupb = groups[iterationNumber-1]
            sameGroupFlag = (iterationNumber-1) == i
            newgroup = Tree()
            newgroup.k = not (groupa.k ^ groupb.k)
            for a in groupa:
                for b in groupb:
                    if sameGroupFlag & (a.v > b.v):
                        continue
                    elif a.v != b.v:
                        CommutatorCounter += 1
                        p = commutator(a.v, b.v, n)
                        if p != 0:
                            searchCounter += len(groups)
                            if sum([group.inTree(p) for group in groups]):
                                #If the element is already in the group lower order groups, don't add it, we can just skip it
                                pass
                            else:
                                searchCounter += 1
                                newgroup.add(p)
                            #If not, then try to add it to the new group
                            #print(newgroup.length)
            #print('Searches, Commutators, NewGroup length')
            #print(searchCounter)
            #print(CommutatorCounter)
            #print(newgroup.length)
            #Add the new group after each [i,j] iteration
            groups.append(newgroup)
            #Add new terms to the h's if we are in an m set
            if newgroup.k == 0:
                for p in newgroup:
                    for h in h_list:
                        for q in h:
                            c = 0
                            c = commutator(p.v, q.v, n)
                            if c != 0:
                                break
                        if c == 0:
                            h.add(p.v)
            #Check if we are done:
            h_len = max([h.length for h in h_list])
            m_len = sum([groups[i].length for i in range(len(groups)) if groups[i].k == 0])
            k_len = sum([groups[i].length for i in range(len(groups)) if groups[i].k == 1])
            #print('h_len, k_len, m_len')
            #print(h_len, k_len, m_len)
            if h_len + k_len == m_len:
                print('Finished using k + h = m')
                return groups
            if newgroup.length == 0:
                end_flag -= 1
                if end_flag < 0:
                    return groups

            
        iterationNumber += 1
        steps -= 1
        print(iterationNumber, [group.length for group in groups], sum([group.length for group in groups]))
    return groups
