import numpy as np
import scipy.linalg as sl
def f(x):
  return 1 + 12*x*x

def u0(x):
  return 1.0 / (1 + 2 * x * x)


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


C = 1.0
T = 2.0
L, R = -5.0, 5.0
#Параметры метода уравнение переноса
n = 40
m = 40
h = (R - L) / n
tau = T / m

#Сетки

x = np.linspace(L,R, n+1)
t = np.linspace(0.0, T, m + 1)
y = np.zeros((m+1, n+1))

#Метод

d = C * tau / h
y[0] = np.vectorize(u0)(x)
for k in range(m):
  for i in range(1, n + 1):
    y[k+1][i] = y[k][i] - d*(y[k][i] - y[k][i-1])

#Точное решение 

def solution(x,t):
  return f(x - C * t)

vsolution = np.vectorize(solution, excluded = ['t'])
u = np.zeros((m+1, n+1))
for k in range(m):
  u[k] = vsolution(x, tau * k)

#Анимация
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')
 

def animate(k):
  plt.clf()
  plt.ylim(0,1)
  plt.title("label",fontname= "" + str(tau * k))
  plt.plot (x , y [ k ] , marker = 'o')
  plt.legend ( " Numerical " )
  plt.plot (x , u [ k ] , marker = '*')
  plt.legend ( " Analytical " )

ani = animation.FuncAnimation( plt.figure (0) , animate , frames = y.shape[0] , interval =100)

ani.save('t.mp4')
