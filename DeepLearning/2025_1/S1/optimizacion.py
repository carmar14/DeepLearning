import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4,10,100)
y= x**3+x**2-4*x+6

plt.plot(x,y,'b')
plt.grid()
plt.show()