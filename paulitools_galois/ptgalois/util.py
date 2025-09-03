import math
import numpy as np
from numpy import kron
from numba import njit
from ptgalois.converter import toString, toZX


def powers_of_two_between(power1, power2):
    if power1 < power2:
        power1, power2 = power2, power1

    log1 = math.log2(power1)
    log2 = math.log2(power2)

    if log1 - log2 > 1:
        result = []
        for i in range(int(log2) + 1, int(log1)):
            result.append(2 ** i)
        return result
    else:
        return None
    

# The signature is critical for the function to be correct
@njit('int_(uint32)')
def popcount(v):
    """ FROM https://stackoverflow.com/questions/71097470/msb-lsb-popcount-in-numba """
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    c = np.uint32((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24
    return c




def getParity(pauliString, basis='Y'):
    """ Returns the parity of a Pauli String

    Parameters
    ----------
    pauliString : str
        A string of Pauli Matrices

    Returns
    -------
    parity : int
        The parity of the Pauli String 0 or 1
    """
    assert basis in ['X', 'Y', 'Z'], "Basis must be one of 'X', 'Y', or 'Z'"
    #Iterates through bitwise with i and i+length//2
    if isinstance(pauliString[0], str):
        strPauli = pauliString
    else:
        strPauli = toString(pauliString)
    assert (len(strPauli) == 1)
    strPauli = strPauli[0]
    parity = strPauli.count(basis)
    return parity % 2


    #for i in range(len(pauliString)):
    #    if pauliString[i] == 'Y' or pauliString[i] == 'Z':
    #        parity += 1
    #return parity % 2

def mergeZXLists(list1, list2):
    """ concatenates two lists of Pauli Strings in the ZX galois form into a single list

    Parameters
    ----------
    list1 : list
        A list of Pauli Strings
    list2 : list
        A list of Pauli Strings

    Returns
    -------
    mergedList : list
        A list of Pauli Strings
    """
    mergedList = []
    for item in list1:
        mergedList.append(item)
    for item in list2:
        mergedList.append(item)
    return mergedList

def pruneGroup(group):
    """ Removes duplicates from a list of Pauli Strings

    Parameters
    ----------
    group : list
        A list of Pauli Strings in the ZX form

    Returns
    -------
    prunedGroup : list
        A list of Pauli Strings
    """
    if isinstance(group[0], str):
        setGroup = set(group)
        prunedGroup = list(setGroup)
        return prunedGroup
        #return group
    else:
        #convert to string:
        strings = toString(group)
        setGroup = set(strings)
        prunedGroup = list(setGroup)
        return toZX(prunedGroup)
    #prunedGroup = []
    #for item in group:
    #    if item not in prunedGroup:
    #        prunedGroup.append(item)
    #return prunedGroup