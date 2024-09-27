from problem.perceptron import Perceptron
from problem.letters import *
import matplotlib.pyplot as plt
import numpy as np


def graph_global_error_in_each_epoch(errors):
    plt.figure(1)
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Global error")
    plt.title("Global error in each epoch")
    plt.grid(True)
    print()
    plt.show()


def main():
    # Get input and expected alphabet
    alphabet = get_alphabet() + get_noisy_alphabet()
    input_alphabet = normalize_input(get_noisy_input_alphabet())
    expected_alphabet = get_noisy_expected_alphabet()

    # Training parameters
    epochs = 5000
    learning_rate = 0.1
    tol = 1

    # Recognize
    recognize_alphabet = Perceptron()
    weights, errors = recognize_alphabet.perceptron_train(
        input_alphabet, expected_alphabet, epochs, learning_rate, tol
    )

    # Graph global error in each epoch
    graph_global_error_in_each_epoch(errors)

    # Predict
    results = recognize_alphabet.predict(input_alphabet, weights)

    # Round to the nearest integer
    rounded_results = np.round(results).astype(int)

    # Print expected results
    print("Expected:")
    for i, letter in enumerate(alphabet):
        # Asumimos que cada 'expected_alphabet' corresponde a un vector asociado a una letra
        print(f"{letter}: {expected_alphabet[i]}", end="\n")

    print("\nResults:")
    for i, letter in enumerate(alphabet):
        # Asumimos que cada 'rounded_results' corresponde a un vector asociado a una letra
        print(f"{letter}: {rounded_results[i]}", end="\n")

    # Check if the results are the same as the expected
    has_errors = False
    if not np.array_equal(expected_alphabet, rounded_results):
        has_errors = True
        wrong_indexes = np.where(expected_alphabet != rounded_results)[0]
        print(
            f"\nPrediction has prediction ERRORS at letter {alphabet[wrong_indexes[0]]}"
        )

    if not has_errors:
        print("\nPrediction is CORRECT")


if __name__ == "__main__":
    main()
