import heapq
from collections import Counter, namedtuple

class HuffmanNode(namedtuple("HuffmanNode", ["char", "freq"])):
    def __lt__(self, other):
        """
        Compare two Huffman nodes based on their frequency.

        Args:
            other (HuffmanNode): The other Huffman node to compare against.

        Returns:
            bool: True if the frequency of this node is less than the other node's frequency, False otherwise.
        """
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    """
    Build a Huffman tree from a given set of frequencies.

    Args:
        frequencies (dict): A dictionary mapping characters to their frequencies.

    Returns:
        HuffmanNode: The root of the Huffman tree.
    """
    # Create a priority queue to store the nodes
    heap = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)

    # Iterate until there is only one node left
    while len(heap) > 1:
        # Get the two nodes with the lowest frequencies
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create a new node with the sum of the frequencies
        merged = HuffmanNode(None, left.freq + right.freq)
        
        # Assign the characters to the new node
        merged.left = left
        merged.right = right
        
        # Add the new node to the priority queue
        heapq.heappush(heap, merged)

    return heap[0]  # Return the root of the Huffman tree

def generate_codes(node, prefix="", code_map={}):
    """
    Generate Huffman codes for a given Huffman tree.

    Args:
        node (HuffmanNode): The root of the Huffman tree.
        prefix (str, optional): The prefix to use for the codes. Defaults to an empty string.
        code_map (dict, optional): The dictionary to store the generated codes in. Defaults to an empty dictionary.

    Returns:
        dict: The dictionary containing the generated codes.
    """
    if node is not None:
        # If the node is a leaf node, add its character and code to the code map
        if node.char is not None:
            code_map[node.char] = prefix
        else:
            # Recursively generate codes for the left and right subtrees
            generate_codes(node.left, prefix + "0", code_map)
            generate_codes(node.right, prefix + "1", code_map)
            
    return code_map

def huffman_encoding(data):
    """
    Encode a given string using Huffman coding.

    Args:
        data (str): The string to encode.

    Returns:
        tuple: A tuple containing the encoded data and the Huffman codes used to encode it.

    """
    # Count the frequency of each character
    frequencies = Counter(data)
    
    # Build the Huffman tree
    root = build_huffman_tree(frequencies)
    
    # Generate Huffman codes
    codes = generate_codes(root)
    
    # Encode the data
    encoded_data = "".join(codes[char] for char in data)
    
    return encoded_data, codes

def huffman_decoding(encoded_data, codes):
    """
    Decode a given encoded string using Huffman coding.

    Args:
        encoded_data (str): The encoded string to decode.
        codes (dict): A dictionary mapping characters to their Huffman codes.

    Returns:
        str: The original decoded string.
    """
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_output = ""
    current_code = ""
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_output += reverse_codes[current_code]
            current_code = ""
    return decoded_output

def main():
    """
    Test the Huffman coding functions with a sample string.

    Encodes and decodes the string "huffman coding is fun!" and prints the original, encoded, and decoded data to the console.
    """
    data = "huffman coding is fun!"
    print("Datos originales:", data)
    
    # Encode
    encoded_data, codes = huffman_encoding(data)
    print("Datos codificados:", encoded_data)
    print("Códigos de Huffman:", codes)
    
    # Decode
    decoded_data = huffman_decoding(encoded_data, codes)
    print("Datos decodificados:", decoded_data)

if __name__ == "__main__":
    main()
