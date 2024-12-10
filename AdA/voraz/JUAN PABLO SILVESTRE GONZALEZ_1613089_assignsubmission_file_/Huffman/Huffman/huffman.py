import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # for priority queue (frecuency basically)
    def __lt__(self, other):
        return self.freq < other.freq

def huffman(frequencies):
    # Priority queue (min-heap)
    heap = [Node(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        # get 2 nodes with the smallest frequencies
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
	
    	# new node, the sum of the 2 nodes selected
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        # new node, so append it into the heap
        heapq.heappush(heap, merged)

    # final node = root of the huffman tree
    root = heap[0]

    # generating the codes
    
    codes = {}
    def generate_codes(node, current_code = ""):
        if node is None:
            return
        if node.char is not None:  # leaf node
            codes[node.char] = current_code
            return
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")

    generate_codes(root)
    return codes
def main():
    frequencies = {
        'e': 20,
        'o': 20,
        'h': 20,
        'l': 40,
    }
    
    codes = huffman(frequencies)

    for i, j in codes.items():
        
        print(f"{i}: {j}")

main()