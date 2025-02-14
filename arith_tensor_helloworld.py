import numpy as np
from scipy import integrate
### Eventually all these tensor need to be written in quimb or itensor

A = np.zeros((2,2,2))
B = np.zeros((2,2,2))
ita = np.nditer(A, flags=['multi_index'])
itb = np.nditer(B, flags=['multi_index'])

def fx(x):
    return x

def gx(x):
    return x*x

xarr = [0,0.25,0.5,0.75,1.0]
yarr = [4,4.5,5,5.5, 6]

dx = [0.125, 0.25,0.25,0.25,0.125]
#xarr = np.arange(0,1,dx)
wp = np.polynomial.laguerre.laggauss(len(xarr))[1]
print(wp)
fxarr = [ fx(xarr[i]) for i in range(len(xarr))]
gyarr = [ gx(yarr[i]) for i in range(len(yarr))]

y = [fxarr[1]+gyarr[i] for i in range(len(yarr))]
## With two variables it should be a grid/mesh
print(y)
# int_simps = integrate.simpson(y,xarr)
# print('integration: ',int_simps)


F1 = np.ones((len(fxarr), 2))
F1[:,1] = [ fxarr[i] for i in range(len(xarr))]
print('test int 1',np.einsum('xa->a',F1))
G1 = np.ones((len(gyarr), 2))
G1[:,1] = gyarr # [ gxarr[i]*dx[i] for i in range(len(gxarr))]

COPY = np.zeros((len(xarr), len(xarr), len(xarr)))
itc = np.nditer(COPY, flags=['multi_index'])
#print([fxarr[i]*gxarr[i] for i in range(len(xarr))])

#### Tensor for addition
for j in ita:
    alpha = ita.multi_index[0]
    beta = ita.multi_index[1]
    gamma = ita.multi_index[2]
    if  (alpha + beta) == gamma:
        A[alpha,beta,gamma] = 1
#### Tensor for multiplication
for j in itb:
    alpha = itb.multi_index[0]
    beta = itb.multi_index[1]
    gamma = itb.multi_index[2]
    if alpha == beta == gamma:
        B[alpha,beta,gamma] =1

#### Tensor for variable copy
for j in itc:
    x = itc.multi_index[0]
    y = itc.multi_index[1]
    z = itc.multi_index[2]

    if x == y == z:
        COPY[x,y,z] = 1

### f(x) + g(x) = F1 G1 A ###
### Perform a tensor contraction
sol = np.einsum('xa, yb, abg -> xyg',F1,G1,A)
print(sol)
# print('test int 1',np.einsum('xa->a',sol))
# Great you have just done your first arithemtic tensor network,
# Next move to multi-valued tensors
print("\n Copy tensor\n")
sol2 = np.einsum('xyz, ya, zb -> xab',COPY,F1,G1)

#print(sol2)
print(COPY)
print(np.einsum('i, ijk -> jk', xarr, COPY))