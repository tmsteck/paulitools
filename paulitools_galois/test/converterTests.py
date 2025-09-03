import unittest
import paulitools.core as ptcore
import numpy as np
import paulitools.converter as ptconv

p_list1 = [[0,1,2], [0,0,0], [1,3,2]]
p_list2 = [['I','X','Y'], ['I','I','I'], ['X','Z','Y']]
p_list3 = ['IXY', 'III', 'XZY']
p_list4 = [[('I', 0), ('X', 1), ('Y', 2)], [('I', 0), ('I', 1), ('I', 2)], [('X', 0), ('Z', 1), ('Y', 2)]] #These are also good formats, as they allow for locality better
p_list5 = [[(0,0), (1,1), (2,2)], [(0,0), (0,1), (0,2)], [(1,0), (3,1), (2,2)]]
p_list6 = []
p_list6 = ['110100', '000000', '101110']
p_listZX = np.array([int('110100',2), int('0',2), int('101110',2)]) 
p_lists = [p_list1, p_list2, p_list3, p_list4, p_list5, p_list6]
#IXY, III, XZY with standard with rightmost index 0 Z component



class coreTests(unittest.TestCase):
    def 