
import torch
import torch.nn.functional as F
import numpy as np


class PathIndex:
    def __init__(self, radius, default_size):
        self.radius = radius
        self.radius_floor = int(np.ceil(radius) - 1)

        self.search_paths, self.search_dst = self.get_search_paths_dst(self.radius)
        self.path_indices, self.src_indices, self.dst_indices = self.get_path_indices(default_size)

    def get_search_paths_dst(self, max_radius=5):
        coord_indices_by_length = [[] for _ in range(max_radius * 4)]

        search_dirs = []
        for x in range(1, max_radius):
            search_dirs.append((0, x))

        for y in range(1, max_radius):
            for x in range(-max_radius + 1, max_radius):
                if x * x + y * y < max_radius ** 2:
                    search_dirs.append((y, x))

        for dir in search_dirs:
            length_sq = dir[0] ** 2 + dir[1] ** 2
            path_coords = []

            min_y, max_y = sorted((0, dir[0]))
            min_x, max_x = sorted((0, dir[1]))

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):

                    dist_sq = (dir[0] * x - dir[1] * y) ** 2 / length_sq

                    if dist_sq < 1:
                        path_coords.append([y, x])

            path_coords.sort(key=lambda x: -abs(x[0]) - abs(x[1]))
            path_length = len(path_coords)

            coord_indices_by_length[path_length].append(path_coords)

        path_list_by_length = [np.asarray(v) for v in coord_indices_by_length if v]
        path_destinations = np.concatenate([p[:, 0] for p in path_list_by_length], axis=0)

        return path_list_by_length, path_destinations

    def get_path_indices(self, size):
        full_indices = np.reshape(np.arange(0, size[0] * size[1], dtype=np.int64), (size[0], size[1]))

        cropped_height = size[0] - self.radius_floor
        cropped_width = size[1] - 2 * self.radius_floor

        path_indices = []
        for paths in self.search_paths:

            path_indices_list = []
            for p in paths:
                coord_indices_list = []

                for dy, dx in p:
                    coord_indices = full_indices[dy:dy + cropped_height,
                                    self.radius_floor + dx:self.radius_floor + dx + cropped_width]
                    coord_indices = np.reshape(coord_indices, [-1])

                    coord_indices_list.append(coord_indices)

                path_indices_list.append(coord_indices_list)

            path_indices.append(np.array(path_indices_list))

        src_indices = np.reshape(full_indices[:cropped_height, self.radius_floor:self.radius_floor + cropped_width], -1)
        dst_indices = np.concatenate([p[:,0] for p in path_indices], axis=0)

        return path_indices, src_indices, dst_indices

