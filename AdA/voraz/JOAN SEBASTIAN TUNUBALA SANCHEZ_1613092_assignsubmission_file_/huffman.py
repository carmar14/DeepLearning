import heapq

def huffman(C):
    n = len(C)
    Q = [(freq, char) for char, freq in C]  # Crear lista de tuplas (frecuencia, carácter)
    heapq.heapify(Q)  # Convertir Q en una cola de prioridad (min-heap)

    for _ in range(1, n):
        z = {}  # Crear un nuevo nodo z

        # Extraer los dos nodos con menor frecuencia
        z['left'] = x = heapq.heappop(Q)  # Nodo con menor frecuencia
        z['right'] = y = heapq.heappop(Q)  # Siguiente menor frecuencia

        # Sumar frecuencias para el nuevo nodo
        z['freq'] = x[0] + y[0]

        # Insertar el nuevo nodo z en Q
        heapq.heappush(Q, (z['freq'], z))

    # Retornar el nodo raíz del árbol
    return heapq.heappop(Q)[1]

C = [('a', 45), ('b', 13), ('c', 12), ('d', 16), ('e', 9), ('f', 5)]
root = huffman(C)

# Función para imprimir el árbol de Huffman (opcional)
def print_huffman_tree(node, prefix=""):
    if isinstance(node, str):  
        print(f"'{node}': {prefix}")
    elif isinstance(node, dict):  
        print_huffman_tree(node['left'][1], prefix + "0")
        print_huffman_tree(node['right'][1], prefix + "1")

print_huffman_tree(root)