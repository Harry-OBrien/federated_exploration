import enum
from turtle import shape
from sklearn.cluster import KMeans
import numpy as np
# from ..data_structures.priority_queue import DPQ_t
import heapq

class Goal_Extractor:
    """
    The Goal Extraction submodule utilizes 𝓂𝑖, and 𝑥𝑖, to cluster frontier locations into four candidates, 𝒢= [g1, g2, g3, g4], 
    where the Euclidean distance between each candidate, g ∈ 𝑋, is maximized to provide a set of spatially distributed goals 
    for a robot to choose from. 𝒢 is updated on- the-fly to account for new exploration goals based on the explored map at each 
    timestep, 𝜏⃗.
    """

    def __init__(self, frontier_width):
        self._frontier_width = frontier_width

    def generate_goals(self, maps):

        # Get frontier indices
        (_, frontier_idxs) = self._generate_frontier_map(self._frontier_width, maps["explored"], maps["obstacles"])

        # cluster points into 4 groups
        clusters, centroids = self._cluster(4, frontier_idxs)
        
        # choose 1 point from each cluster that are as far as possible from each other
        return self._select_goal_locations(clusters, centroids)

    def _frontier_for_region(self, region, frontier_map, explored, obstacles):
        frontier_indices = ([], [])
        map_width, map_height = frontier_map.shape
        
        for i in range(len(region[0])):
            x, y = (region[0, i], region[1, i])

            offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            for offset in offsets:
                x_frontier, y_frontier = (x + offset[0], y + offset[1])

                # Check if we're currently on an obstacle or idx is out of bounds
                if obstacles[x,y]\
                    or x_frontier < 0 or x_frontier >= map_width\
                    or y_frontier < 0 or y_frontier >= map_height:
                    continue

                # Check if this is a valid point for the frontier
                if not frontier_map[x_frontier, y_frontier] and\
                    not explored[x_frontier, y_frontier] and\
                    not obstacles[x_frontier, y_frontier]:
                    frontier_indices[0].append(x_frontier)
                    frontier_indices[1].append(y_frontier)

        return frontier_indices

    def _generate_frontier_map(self, frontier_width, explored, obstacles):
        frontier_map = np.zeros_like(explored)
        frontier_idxs = []

        # find boundary
        explored_indices = np.where(explored==1)

        # repeat for n steps to define border width
        for _ in range(frontier_width):
            explored_indices = self._frontier_for_region(np.array(explored_indices), frontier_map, explored, obstacles)
            frontier_idxs.extend(zip(explored_indices[0], explored_indices[1]))
            
            for i in range(len(explored_indices[0])):
                frontier_map[explored_indices[0][i], explored_indices[1][i]] = 1

        return (frontier_map, frontier_idxs)
                            
    def _cluster(self, n_clusters, points):
        """
        Clusters the points given into k clusters

        # Parameters
            n_cluster: Int - the number of clusters to group points into
            points: [(Int, Int)] - the idx of the points given (in this case, it will typically be the frontier points)

        # Return
            [(Int, Int)] 2d array of points (row, col)
        """
        # Cluster points
        km = KMeans(
            n_clusters=n_clusters, init='k-means++',
            n_init=3, max_iter=50,
            tol=1e-04, random_state=0
        )

        predictions = km.fit_predict(points)
        clusters = [[] for _ in range(n_clusters)]

        for i, idx in enumerate(predictions):
            clusters[idx].append(points[i])

        return clusters, km.cluster_centers_

    def _goal_fitness(self, points):
        # wanting to maximise dist
        dist = 0
        for i, pos in enumerate(points):
            for j in range(i+1, 4):
                dist += np.sqrt((points[j][0] - pos[0])**2 + (points[j][1] - pos[1])**2)

        return dist

    def _select_goal_locations(self, clusters, centroids):
        best_idxs = [0] * 4

        for i, cluster in enumerate(clusters):
            assert(len(cluster) > 0)
            best_score = 0
            for j, point in enumerate(cluster):
                new_points = centroids
                new_points[i] = point
                new_score = self._goal_fitness(new_points)

                if new_score > best_score:
                    best_score = new_score
                    best_idxs[i] = j

        return [clusters[i][pos] for i, pos in enumerate(best_idxs)]

from matplotlib import pyplot as plt
def show_image(img, vmax=1):
    
    plt.imshow(img, cmap='gray', vmin=0, vmax=vmax)
    plt.axis('off')
    plt.show()

# if __name__ == "__main__":
#     goal_generator = Goal_Extractor(frontier_width=2)
#     maps = {}

#     # maps["explored"] = np.array([
#     #     [0, 0, 0, 0, 0, 0, 0, 0],
#     #     [0, 0, 0, 1, 1, 1, 1, 0],
#     #     [0, 0, 0, 1, 1, 1, 1, 0],
#     #     [0, 0, 0, 1, 1, 1, 1, 0],
#     #     [0, 0, 0, 1, 1, 1, 0, 0],
#     #     [0, 0, 1, 1, 1, 1, 0, 0],
#     #     [0, 0, 1, 1, 1, 1, 0, 0],
#     #     [0, 0, 1, 1, 1, 1, 0, 0]])

#     # maps["obstacles"] = np.array([
#     #     [0, 0, 0, 0, 0, 0, 0, 0],
#     #     [0, 0, 0, 1, 1, 1, 1, 0],
#     #     [0, 0, 0, 0, 0, 0, 1, 0],
#     #     [0, 0, 0, 0, 0, 1, 1, 0],
#     #     [0, 0, 0, 0, 0, 0, 0, 0],
#     #     [0, 0, 0, 0, 0, 0, 0, 0],
#     #     [0, 0, 1, 0, 0, 1, 0, 0],
#     #     [0, 0, 1, 1, 0, 0, 0, 0]])

#     maps["explored"] = np.array([
#         [0, 0, 0, 1, 1, 0, 0, 0],
#         [0, 0, 0, 1, 1, 0, 0, 0],
#         [0, 0, 1, 1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1],
#         [0, 0, 1, 1, 1, 1, 0, 0],
#         [0, 0, 0, 1, 1, 0, 0, 0],
#         [0, 0, 0, 1, 1, 0, 0, 0],])

#     maps["obstacles"] = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0]])

#     goals = goal_generator.generate_goals(maps)
#     print(goals)