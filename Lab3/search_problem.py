import numpy as np
from scipy.spatial.distance import cdist


class AirportProblem:

    def __init__(self, n_cities, n_airports):
        self.n_cities = n_cities
        self.n_airports = n_airports
        self.city_locations = np.random.random((n_cities, 2))
    
    def sum_city_distances(self, location_vector: np.array) -> float:
        """
        computes distances between each city and it's closest airport. Returns the sum of these distances
        :param location_vector: vector containing coordinates of all airports: [xo, y0, x1, y1, ... xn, yn]
        """
        assert len(location_vector) == self.n_airports * 2

        location_vector = np.array(location_vector).reshape((-1, 2))
        dists = cdist(location_vector, self.city_locations, metric='euclidean')
        return dists.min(axis=0).sum()
