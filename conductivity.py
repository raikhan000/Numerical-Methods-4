import numpy as np
import scipy.linalg as sl
def f(x):
  return 1 + 12*x*x

def u0(x):
  return np.exp(-x*x/4)

n = 10
u = np.array([0] + [0] + [-1] * (n-1))
d = np.array([1] + [2]*(n - 1) + [1])
l = np.array([-1] * (n - 1) + [0] + [0])
A = np.array([u,d,l])

h = 1.0/ n
x = np.linspace(0,1, n+1)

b = h*h* np.vectorize(f)(x)
b[0] = 1
b[n] = 1

y = sl.solve_banded((1,1), np.array([u,d,l]), b)


#Уравнение теплопроводости

mu = 1.0
T = 2.0
L, R = -10.0, 10.0

#параметры метода

n = 40
m = 40
h = (R - L) / n
tau = T / m

#Сетки

x = np.linspace(L,R, n+1)
t = np.linspace(0.0, T, m + 1)
y = np.zeros((m+1,n+1))

#Метод
d = mu * tau / ( h * h )
y [0] = np . vectorize ( u0 )( x )

for k in range(m):
  for i in range(1,n):
    y[k+1][i] = y[k][i] + d*(y[k][i-1] - 2* y[k][i])

def solution(x,t):
  return 1/ np.sqrt(t+1) * np.exp(-x*x/4/(t+1))
vsolution = np.vectorize(solution, excluded=['t'])
u = np.zeros((m+1, n+1))
for k in range(m):
  u[k] = vsolution(x, tau * k)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')
 
 
fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,

def animate(k):
  plt.clf()
  plt.ylim(0,1)
  plt.title("label",fontname= "" + str(tau * k))
  plt.plot (x , y [ k ] , marker = 'o')
  plt.legend ( " Numerical " )
  plt.plot (x , u [ k ] , marker = '*')
  plt.legend ( " Analytical " )
  plt.show ()

ani = animation . FuncAnimation ( plt.figure (0) , animate , frames = y.shape[0] , interval =100)

plt.show ()
#anim.save('bloch_sphere.mp4', fps=20)
ani.save('conductivity.mp4')
