#Copyright 2024 Tomas mancera villa 98649

import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # Definir cómo comparar nodos en la cola de prioridad
    def __lt__(self, other):
        return self.freq < other.freq


def huffman_greedy(char_freqs):

    heap = [Node(char, freq) for char, freq in char_freqs]
    heapq.heapify(heap)


    while len(heap) > 1:
        # Extraer los dos nodos con menor frecuencia (decisión voraz)
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # Crear un nuevo nodo combinando los dos extraídos
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        # Insertar el nuevo nodo en la cola de prioridad
        heapq.heappush(heap, merged)

    # El último nodo es la raíz del árbol de Huffman
    return heapq.heappop(heap)

# Función para generar los códigos de Huffman
def generate_huffman_codes(root, current_code="", codes={}):
    if root is None:
        return

    # Si es una hoja, asignar el código actual
    if root.char is not None:
        codes[root.char] = current_code
        return

    # Recorrer el árbol (decisión recursiva basada en la estructura del árbol)
    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)

    return codes


if __name__ == "__main__":
    # Lista de caracteres con sus frecuencias
    char_freqs = [
        ('a', 45),
        ('b', 13),
        ('c', 12),
        ('d', 16),
        ('e', 9),
        ('f', 5)
    ]

    # Construir el árbol de Huffman usando el enfoque voraz
    huffman_tree = huffman_greedy(char_freqs)

    # Generar los códigos de Huffman
    codes = generate_huffman_codes(huffman_tree)

    # Imprimir los códigos de Huffman generados
    print("Códigos de Huffman:")
    for char, code in codes.items():
        print(f"{char}: {code}")
