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
            # sns.heatmap(js_matrix, annot=False, cmap='coolwarm')
            cluster_grid = sns.clustermap(
                            js_matrix,
                            cmap='coolwarm',
                            metric='euclidean',
                            method='average',
                            figsize=(10, 8),
                            linewidths=0.5,
                            square=True,  
                            cbar_kws={
                                'orientation': 'vertical',  
                                'shrink': 0.4,             
                                'aspect': 10,              
                                'label': 'JS Distance'
                            })
            

            cluster_grid.ax_heatmap.set_title("Jensen-Shannon Distance Matrix", fontsize=14, fontweight='bold')
            cluster_grid.ax_heatmap.set_xlabel("Individual", fontsize=12)
            cluster_grid.ax_heatmap.set_ylabel("Individual", fontsize=12)
            
            # cluster_grid.cax.set_position([0.91, 0.3, 0.02, 0.4])

            for line in cluster_grid.ax_col_dendrogram.collections:
                line.set_linewidth(0.0)
            for line in cluster_grid.ax_row_dendrogram.collections:
                line.set_linewidth(2.5)

        return js_matrix
    
    
    
    def inner_product_heatmap(self, show=True):
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt

        n = self.struggle_vector.shape[0]
        similarity_matrix = np.zeros((n, n))

        # normalization
        odd_norm = np.array([self._normalize_and_binning(v) for v in self.odd_struggle_vector])
        even_norm = np.array([self._normalize_and_binning(v) for v in self.even_struggle_vector])

        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = np.dot(odd_norm[i], even_norm[i])
                else:
                    sim1 = np.dot(odd_norm[i], odd_norm[j])
                    sim2 = np.dot(even_norm[i], even_norm[j])
                    similarity_matrix[i, j] = 0.5 * (sim1 + sim2)

        if show:
            # 创建聚类热图对象
            cluster_grid = sns.clustermap(
                similarity_matrix,
                cmap='coolwarm',
                metric='euclidean',
                method='average',
                figsize=(10, 8),
                linewidths=0.5,
                square=True,  # ✅ 每个格子为正方形
                cbar_kws={
                    'orientation': 'vertical',  # ✅ colorbar 放右边（默认是）
                    'shrink': 0.4,              # ✅ 控制长度
                    'aspect': 10,               # ✅ 控制细长比例
                    'label': 'Inner Product Similarity'
                }
            )

            # 设置标题、标签
            cluster_grid.ax_heatmap.set_title("Inner Product Similarity Matrix", fontsize=14, fontweight='bold')
            cluster_grid.ax_heatmap.set_xlabel("Individual", fontsize=12)
            cluster_grid.ax_heatmap.set_ylabel("Individual", fontsize=12)

            # ✅ 加粗 dendrogram 连线
            for line in cluster_grid.ax_col_dendrogram.collections:
                line.set_linewidth(0.0)
            for line in cluster_grid.ax_row_dendrogram.collections:
                line.set_linewidth(2.5)

            plt.show()

        return similarity_matrix

