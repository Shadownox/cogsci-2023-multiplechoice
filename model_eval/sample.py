import numpy as np

# Calculation in the example:
# Ground truth: 2 responses are given
# accuracy for a model with just one prediction:
print("Example: Accuracy with 1 prediction:",
    2/9 * 8/9 +
    7/9 * 6/9
)

# accuracy for a model with just two prediction:
print("Example: Accuracy with 2 predictions:", 
      2/9 * 1/8 * 1 +
      2/9 * 7/8 * 7/9 +
      7/9 * 2/8 * 7/9 +
      7/9 * 6/8 * 5/9)


print()
print("Comparison between Jaccard and Accuracy with sampling:")

for num in range(1, 10):
    print("Number of predictions:", num)
    target = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])

    pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(num):
        pred[i] = 1

    accs = []
    jaccs = []
    for i in range(500000):
        rand_pred = np.random.permutation(pred)
        acc = np.sum(rand_pred == target) / 9
        
        jacc_int = 0
        jacc_union = 0
        for j in range(9):
            if target[j] == 1 and rand_pred[j] == 1:
                jacc_int += 1
            if target[j] == 1 or rand_pred[j] == 1:
                jacc_union += 1

        jaccs.append(jacc_int / jacc_union)
        accs.append(acc)

    print("Accuracy:", np.mean(accs))
    print("Jaccard:", np.mean(jaccs))
    print()
