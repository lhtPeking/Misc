import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

class Similarity:
    def __init__(self, struggle_vector, bins=320):
        self.struggle_vector = struggle_vector
        self.half_length = self.struggle_vector.shape[1] // 2
        
        self.odd_struggle_vector = self.struggle_vector[:, :self.half_length]
        self.even_struggle_vector = self.struggle_vector[:, self.half_length:]
        
        self.bins = bins
        
    def _normalize_and_binning(self, vec, epsilon=1e-12):
        vec = np.asarray(vec)
        min_val, max_val = np.min(vec), np.max(vec)
        value_range = (min_val, max_val)

        hist, _ = np.histogram(vec, bins=self.bins, range=value_range, density=False)

        hist = hist + epsilon
        return hist / np.sum(hist)
    
    def JS_heatmap(self, show=True):
        n = self.struggle_vector.shape[0]
        js_matrix = np.zeros((n, n))

        # normalization
        odd_norm = np.array([self._normalize_and_binning(v) for v in self.odd_struggle_vector])
        even_norm = np.array([self._normalize_and_binning(v) for v in self.even_struggle_vector])

        for i in range(n):
            for j in range(n):
                if i == j:
                    js_matrix[i, j] = jensenshannon(odd_norm[i], even_norm[i])
                else:
                    d1 = jensenshannon(odd_norm[i], odd_norm[j])
                    d2 = jensenshannon(even_norm[i], even_norm[j])
                    js_matrix[i, j] = 0.5 * (d1 + d2)

        if show:
            plt.figure(figsize=(8, 6))
            sns.heatmap(js_matrix, annot=False, cmap='viridis')
            plt.title("Jensen-Shannon Distance Matrix")
            plt.xlabel("Individual")
            plt.ylabel("Individual")
            plt.tight_layout()
            plt.show()

        return js_matrix
    
    def inner_product_heatmap(self, show=True):
        n = self.struggle_vector.shape[0]
        similarity_matrix = np.zeros((n, n))

        # normalization
        odd_norm = np.array([self._normalize(v) for v in self.odd_struggle_vector])
        even_norm = np.array([self._normalize(v) for v in self.even_struggle_vector])

        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = np.dot(odd_norm[i], even_norm[i])
                else:
                    sim1 = np.dot(odd_norm[i], odd_norm[j])
                    sim2 = np.dot(even_norm[i], even_norm[j])
                    similarity_matrix[i, j] = 0.5 * (sim1 + sim2)

        if show:
            plt.figure(figsize=(8, 6))
            sns.heatmap(similarity_matrix, annot=False, cmap='plasma')
            plt.title("Inner Product Similarity Matrix")
            plt.xlabel("Individual")
            plt.ylabel("Individual")
            plt.tight_layout()
            plt.show()

        return similarity_matrix
