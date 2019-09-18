assert False
# https://stackoverflow.com/a/39395973
# https://www.ianlewis.org/en/mixins-and-python
# https://www.python.org/download/releases/2.3/mro/

O = object

class F(O): pass

class E(O): pass

class D(O): pass

class C(D, F): pass

class B(E, D): pass

class A(B, C): pass


# Linearizations with left-to-right depth-first order
L(F) = FO
L(E) = EO
L(D) = DO
L(C) = C + {L(D), L(F), DF} = C + {DO, FO, DF} = CDFO
L(B) = B + {L(E), L(D), ED} = B + {EO, DO, ED} = BEDO
L(A) = A + {L(B), L(C), BC} = A + {BEDO, CDFO, BC} = ABECDFO

# take the head of the first list, i.e L[B1][0]; if this head 
# is not in the tail of any of the other lists, then add it to
# the linearization of C and remove it from the lists in the merge,
# otherwise look at the head of the next list and take it, if
# it is a good head. Then repeat the operation until all the
# class are removed or it is impossible to find good heads. In
# this case, it is impossible to construct the merge, Python
# 2.3 will refuse to create the class C and will raise an exception.

print("".join(cls.__name__ for cls in A.mro()))