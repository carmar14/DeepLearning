import numpy as np
import matplotlib.pyplot as plt

# Definir la función
def f(x):
    return x**4 - 4*x**2

# Crear valores de x en un rango adecuado
x = np.linspace(-3, 3, 400)
y = f(x)

# Graficar la función
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$f(x) = x^4 - 4x^2$", color="b")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")

# Marcar los puntos críticos
critical_points = [0, np.sqrt(2), -np.sqrt(2)]
plt.scatter(critical_points, [f(c) for c in critical_points], color="red", zorder=3, label="Mínimos y máximo")

# Etiquetas y leyenda
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gráfica de $f(x) = x^4 - 4x^2$")
plt.legend()
plt.grid(True)
plt.show()
