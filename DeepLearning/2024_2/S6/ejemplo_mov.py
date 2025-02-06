import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del proyectil
v0 = 50  # Velocidad inicial en m/s
theta = 30  # Ángulo de lanzamiento en el plano XY en grados
phi = 45  # Ángulo de lanzamiento en el eje Z en grados
g = 9.81  # Aceleración debida a la gravedad en m/s^2
k = 0.1  # Coeficiente de resistencia del aire
m = 1  # Masa del proyectil en kg

# Convertir ángulos de grados a radianes
theta_rad = np.radians(theta)
phi_rad = np.radians(phi)

# Calcular las componentes de la velocidad inicial
v0x = v0 * np.cos(theta_rad) * np.cos(phi_rad)
v0y = v0 * np.sin(theta_rad) * np.cos(phi_rad)
v0z = v0 * np.sin(phi_rad)

# Tiempo de simulación
t_max = 5  # Tiempo máximo en segundos
dt = 0.1  # Incremento de tiempo
t = np.arange(0, t_max, dt)

# Trajectoria sin resistencia del aire
x_no_resistencia = v0x * t
y_no_resistencia = v0y * t
z_no_resistencia = v0z * t - 0.5 * g * t**2

# Trajectoria con resistencia del aire (numérica)
x_con_resistencia = []
y_con_resistencia = []
z_con_resistencia = []

# Inicializar condiciones
vx = v0x
vy = v0y
vz = v0z

# Simulación numérica para resistencia del aire
for time in t:
    x_con_resistencia.append(vx * time)
    y_con_resistencia.append(vy * time)
    vz = vz - g * dt  # Actualizar vz con gravedad
    vx *= np.exp(-k / m * dt)  # Actualizar vx con resistencia
    vy *= np.exp(-k / m * dt)  # Actualizar vy con resistencia
    z_con_resistencia.append(vz * time - 0.5 * g * time**2)

# Convertir listas a arrays
x_con_resistencia = np.array(x_con_resistencia)
y_con_resistencia = np.array(y_con_resistencia)
z_con_resistencia = np.array(z_con_resistencia)

# Graficar las trayectorias
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Trayectoria sin resistencia
ax.plot(x_no_resistencia, y_no_resistencia, z_no_resistencia, label='Sin Resistencia', color='blue')

# Trayectoria con resistencia
ax.plot(x_con_resistencia, y_con_resistencia, z_con_resistencia, label='Con Resistencia', color='red')

# Dibujar el vector de velocidad inicial
ax.quiver(0, 0, 0, v0x, v0y, v0z, color='green', label=r'$\vec{v_0}$', arrow_length_ratio=0.1)

# Dibujar ángulos
# Ángulo theta (en el plano XY)
ax.quiver(0, 0, 0, v0 * np.cos(theta_rad), v0 * np.sin(theta_rad), 0, color='purple', label=r'$\theta$', linestyle='dotted', arrow_length_ratio=0.1)
# Ángulo phi (en el eje Z)
ax.quiver(0, 0, 0, 0, 0, v0z, color='orange', label=r'$\phi$', linestyle='dotted', arrow_length_ratio=0.1)

# Configurar límites de los ejes
ax.set_xlim([0, 200])
ax.set_ylim([0, 200])
ax.set_zlim([0, 150])

# Etiquetas y título
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Trayectoria de un Proyectil y Vectores de Velocidad Inicial')
ax.legend()

# Mostrar el gráfico
plt.show()
