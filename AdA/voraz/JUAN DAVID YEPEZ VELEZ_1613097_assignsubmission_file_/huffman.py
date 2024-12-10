import heapq

# Definición de la clase Nodo
class Node:
    def __init__(self, char, freq):
        self.char = char  # Caracter
        self.freq = freq  # Frecuencia
        self.left = None  # Nodo izquierdo
        self.right = None  # Nodo derecho

    def __lt__(self, other):
        return self.freq < other.freq  # Para que heapq ordene los nodos por frecuencia

# Función para construir el árbol de Huffman
def huffman_tree(frequencies):
    # Crear una cola de prioridad (heap) con nodos de Huffman
    priority_queue = []
    
    # Crear un nodo para cada carácter y agregarlo a la cola
    for char, freq in frequencies.items():
        heapq.heappush(priority_queue, Node(char, freq))
    
    # Construcción del árbol de Huffman
    while len(priority_queue) > 1:
        # Extraer los dos nodos con menor frecuencia
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        # Crear un nodo padre cuya frecuencia es la suma de las frecuencias de los dos nodos extraídos
        merged_node = Node(None, left.freq + right.freq)
        merged_node.left = left
        merged_node.right = right
        
        # Insertar el nodo fusionado de nuevo en la cola
        heapq.heappush(priority_queue, merged_node)
    
    # El único nodo restante es la raíz del árbol de Huffman
    return priority_queue[0]

# Función para generar el código de Huffman
def generate_huffman_codes(node, prefix="", codebook={}):
    if node is not None:
        # Si el nodo es una hoja (un carácter), asignar el código
        if node.char is not None:
            codebook[node.char] = prefix
        # Recursión sobre los hijos izquierdo y derecho
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)
    return codebook

# Ejemplo de uso
if __name__ == "__main__":
    # Frecuencias de los caracteres
    frequencies = {
        'a': 5,
        'b': 9,
        'c': 12,
        'd': 13,
        'e': 16,
        'f': 45
    }
    
    # Construir el árbol de Huffman
    root = huffman_tree(frequencies)
    
    # Generar el código de Huffman
    huffman_codes = generate_huffman_codes(root)
    
    # Mostrar los códigos de Huffman
    print("Códigos de Huffman:")
    for char, code in huffman_codes.items():
        print(f"{char}: {code}")
