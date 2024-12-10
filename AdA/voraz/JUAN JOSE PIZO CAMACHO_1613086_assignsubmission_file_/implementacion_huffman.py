import heapq

def huffman(C):
    # Convierto la lista de frecuencias en un heap (cola de prioridad)
    heapq.heapify(C)
    while len(C) > 1:
        x = heapq.heappop(C)
        y = heapq.heappop(C)
        z = (x[0] + y[0], x, y)
        heapq.heappush(C, z)
    
    return heapq.heappop(C)

def encode_and_decode(frequencies, message):
    # Aqui lo que hace es Construir el árbol de Huffman
    tree = huffman(frequencies)
    
    # Esta es la función interna para generar los códigos a partir del árbol
    def generate_codes(tree, prefix=""):
        if len(tree) == 2:  # Nodo hoja
            return {tree[1]: prefix}
        left = generate_codes(tree[1], prefix + "0")  # Recorremos el lado izquierdo
        right = generate_codes(tree[2], prefix + "1")  # Recorremos el lado derecho
        return {**left, **right}

    # Aqui se genera los códigos de Huffman
    codes = generate_codes(tree)
    
    # Codifico el mensaje
    encoded_message = ''.join(codes[char] for char in message)
    
    # Decodifico el mensaje
    decoded_message = []
    node = tree
    for bit in encoded_message:
        node = node[1] if bit == '0' else node[2]
        if len(node) == 2:  # Si es un nodo hoja, extraemos el carácter
            decoded_message.append(node[1])
            node = tree  # Volvemos a la raíz del árbol
    
    return codes, encoded_message, ''.join(decoded_message)

frequencies = [(5, 'a'), (9, 'b'), (12, 'c'), (13, 'd'), (16, 'e'), (45, 'f')]
message = "abcdef"
codes, encoded, decoded = encode_and_decode(frequencies, message)

print("Códigos de Huffman:", codes)
print("Mensaje codificado:", encoded)
print("Mensaje decodificado:", decoded)
