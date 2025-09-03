from ptgalois.converter import toString, toZX

test_set = ['XXYY', 'IZXY']
test_1_ZX = toZX(test_set[0])
test_2_ZX = toZX(test_set[1])
test_to_str = toString([test_1_ZX[0], test_2_ZX[0]])
test_ZX = toZX(test_set)
test_str = toString(test_ZX)
#print(toString(test_ZX))
#print(test_ZX)
