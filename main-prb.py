from scipy.optimize import *
from math import pi#, sqrt, cos, sin
from cmath import exp
from numpy import real, arange, array, savetxt, zeros,seterr, linalg
from sympy import Matrix, diff,symbols, sqrt, cos, sin
from sympy.utilities.lambdify import lambdify
from sympy.utilities.iterables import flatten
import time

start_time = time.time()

c0 = 6.832
c1 = 4.064
c2 = -0.374
c3 = -0.095
K = 69518.0
G = 47352.0
pn = 4
s = symbols('s[0:32]')
grid = 10
z = pi / 60
a = 2.47
ay = a*sqrt(3)/2
A = Matrix([[0.5,sqrt(3)/2],[sqrt(3)/2,-0.5]])

class vector():
    def __init__(self, x, y):
        self.x = x
        self.y = y
def Ur(bx, by, ui):
    # Ux(px,py)=Ui[pn*py+px]
    # Uy(px,py)=Ui[pn*pn+pn*py+px]
    ret = vector(0, 0)
    x = 0
    y = 0
    for py in range(0, pn):
        for px in range(0, pn):
            e = cos((bx*px/a + by*py/ay)*2*pi)
            x += ui[pn*py+px] * e
            y += ui[pn*pn+pn*py+px] * e
    ret.x = x
    ret.y = y
    return ret

def Einter(bx, by, ui):
    ux = 0
    uy = 0
    if (bx != 0 and by/bx > 1/sqrt(3)) or (bx == 0 and by != 0):
        bxs,bys = A*Matrix([[bx],[by]])
        for py in range(0, pn):
            for px in range(0, pn):
                e = cos((bxs*px/a + bys*py/ay)*2*pi)
                ux += ui[pn*py+px] * e
                uy += ui[pn*pn+pn*py+px] * e
        ux,uy = A*Matrix([[ux],[uy]]) 
    else:
        for py in range(0, pn):
            for px in range(0, pn):
                e = cos((bx*px/a + by*py/ay)*2*pi)
                ux += ui[pn*py+px] * e
                uy += ui[pn*pn+pn*py+px] * e
                    
    bx = bx + 2*ux
    by = by + 2*uy
    v = 2 * pi / a * (bx - 1/sqrt(3) * by)
    w = 2 * pi / a * 2/sqrt(3) * by
    ret = c0
    ret += c1 * (cos(v) + cos(w) + cos(v+w))
    ret += c2 * (cos(v+2*w) + cos(v-w) + cos(2*v+w))
    ret += c3 * (cos(2*v) + cos(2*w) + cos(2*v+2*w))
    return ret


def Eintra(bx, by, ui):
    xx = 0
    xy = 0
    yx = 0
    yy = 0
    if (bx != 0 and by/bx > 1/sqrt(3)) or (bx == 0 and by != 0):
        bxs,bys = A*Matrix([[bx],[by]])    
        for py in range(0, pn):
            for px in range(0, pn):
                e = -sin((bxs*px/a + bys*py/ay)*2*pi)
                xx += ui[pn*py+px] * e * px*2*pi/a
                yx += ui[pn*pn+pn*py+px] * e * px*2*pi/a
                xy += ui[pn*py+px] * e * py*2*pi/ay
                yy += ui[pn*pn+pn*py+px] * e * py*2*pi/ay
        xx,xy,yx,yy = Matrix([[xx,xy],[yx,yy]])*A 
    else:
        for py in range(0, pn):
            for px in range(0, pn):
                e = -sin((bx*px/a + by*py/ay)*2*pi)
                xx += ui[pn*py+px] * e * px*2*pi/a
                yx += ui[pn*pn+pn*py+px] * e * px*2*pi/a
                xy += ui[pn*py+px] * e * py*2*pi/ay
                yy += ui[pn*pn+pn*py+px] * e * py*2*pi/ay
    
    #ret = G*(Uxx+Uyy)**2 + K * ((Uxx-Uyy)**2+(Uxy+Uyx)**2)
    #bx = -zry; by = zrx; Uxx = zuxy; Uxy = -zuxx; Uyx = zuyy; Uyy = -zuyx
    ret = G*(xy-yx)**2 + K * ((xy+yx)**2 + (-xx+yy)**2)
    return ret

def Etotal(ui):
    ret = 0
    for by in arange(0, ay, ay/grid):
        for bx in arange(by/sqrt(3),a + by/sqrt(3), a/grid):
            ret += Eintra(bx, by, ui)
            ret += Einter(bx, by, ui)
    if type(ui) is tuple:
        return ret
    else:
        return ret.evalf(15)
    
def g(ui):
    gra = zeros(2*pn**2)
    for i in range(0, 2*pn**2):
        lam_f = lambdify(flatten(s),e[i])
        gra[i] = lam_f(*flatten(ui))
    return gra

def h(ui):
    Hess = zeros((2*pn**2,2*pn**2))
    for i in range(0, 2*pn**2):
        for j in range(0, 2*pn**2):
            lam_f = lambdify(flatten(s),ee[i][j])
            Hess[i][j] = lam_f(*flatten(ui))
    return Hess

E = Etotal(s)
e = [0]*pn*pn*2
for i in range(0, 2*pn**2):
    e[i] = diff(E,s[i])
ee = [[0 for i in range(pn*pn*2)] for j in range(pn*pn*2)]
for i in range(0, 2*pn**2):
    for j in range(0, 2*pn**2):
        ee[i][j] = diff(E,s[i],s[j])
    
def test():
    u = array([0]*pn*pn*2)
    print(Etotal(u))
    Hess = h(u)
    Hess = Hess + Hess.T
    try:
        Test = linalg.cholesky(Hess)
        print('positive definite Hessian')
    except:
        print('Non-positive definite Hessian')
        return
    tt = []
    ans = fmin_l_bfgs_b(Etotal, u, fprime=g)
    print(ans)
    # for by in arange(0, ay, ay/grid):
        # for bx in arange(by/sqrt(3),a + by/sqrt(3), a/grid):
            # ur = Ur(bx, by, ans)
            # tt += [[bx, by, ur.x, ur.y]]
            
    # soa = array(tt)
    # savetxt('./soat.txt', soa)
    return
    
test()

print("Time taken: %2.2f s"%(time.time() - start_time))
