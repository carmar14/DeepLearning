class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

class HuffmanCoding:
    def __init__(self, frequencies):
        self.frequencies = frequencies
        self.codes = {}
        self.root = None

    def build_tree(self):
        # Crear una lista de nodos
        nodes = [Node(char, freq) for char, freq in self.frequencies.items()]

        # Construir el árbol de Huffman
        while len(nodes) > 1:
            # Ordenar nodos por frecuencia
            nodes.sort(key=lambda x: x.freq)

            # Tomar los dos nodos de menor frecuencia
            left = nodes.pop(0)
            right = nodes.pop(0)

            # Crear un nodo combinado
            merged = Node(None, left.freq + right.freq)
            merged.left = left
            merged.right = right

            # Añadir el nuevo nodo a la lista
            nodes.append(merged)

        # El último nodo es la raíz del árbol
        self.root = nodes[0]

    def generate_codes(self, node=None, current_code=""):
        if node is None:
            node = self.root

        # Si es una hoja, asignar el código al carácter
        if node.char is not None:
            self.codes[node.char] = current_code
            return

        # Recorrer los subárboles izquierdo y derecho
        self.generate_codes(node.left, current_code + "0")
        self.generate_codes(node.right, current_code + "1")

    def get_codes(self):
        if not self.codes:
            self.generate_codes()
        return self.codes

# Ejemplo de uso
if __name__ == "__main__":
    frequencies = {
        'a': 5,
        'b': 9,
        'c': 12,
        'd': 13,
        'e': 16,
        'f': 45
    }

    huffman = HuffmanCoding(frequencies)
    huffman.build_tree()
    codes = huffman.get_codes()

    print("Códigos de Huffman:")
    for char, code in codes.items():
        print(f"{char}: {code}")
