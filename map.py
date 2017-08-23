#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from datetime import datetime

import voronoi
import perlin


LAND = 0.25
WATER = 0.15
OCEAN = 0


def dist(a, b):
    return np.linalg.norm(np.array(a) - b)


def getVoronoi(size, bbox):
    print("Loading Voronoi cells...")

    centers, neighbors, vertices, regions = np.load('voronoi32k.npy')

    # Generate voronoi32k.npy once, load it every other time

    # centers, neighbors, vertices, regions = \
    #         voronoi.relaxed_voronoi(32000, bbox)
    # np.save('voronoi32k.npy', (centers, neighbors, vertices, regions))


    vert_neighbors = [[] for i in vertices]
    vert_regions = [[] for i in vertices]
    for r, p in enumerate(regions):
        for i, v in enumerate(p[:-1]):
            vert_regions[v].append(r)
            if p[i-1] not in vert_neighbors[v]:
                vert_neighbors[v].append(p[i-1])
            if p[i+1] not in vert_neighbors[v]:
                vert_neighbors[v].append(p[i+1])

    return centers, neighbors, vertices, regions


def getBasicShape(size, neighbors, vertices, regions, seed=None):
    if seed is None:
        seed = datetime.now().microsecond

    print("Calculating landmass shape...")

    print("Landmass shape seed: %d" % seed)
    np.random.seed(seed)

    noise = perlin.Perlin(size, 8)

    cell_type = []

    def isLand(c):
        v = [int(t) for t in c]

        x = 2. * c[0] / size - 1
        y = 2. * c[1] / size - 1

        return noise[v[0], v[1]] > 0.316 * (1 + x**2 + y**2)

    for r in regions:
        land = 0
        thresh = 0.7 * len(r)
        for v in vertices[r]:
            if np.any([v == 0, v == size]):
                land = 0
                break
            if dist(v, (1756, 666)) < 25:
                land = 0
                break
            land += 1 if isLand(v) else 0
        cell_type.append(LAND if land > thresh else WATER)

    corner = np.where(np.all(vertices == [0,0], axis=1))[0][0]

    flood = [corner]

    cur_index = 0

    while cur_index < len(flood):
        current = flood[cur_index]
        cell_type[current] = OCEAN
        cur_index += 1

        for n in neighbors[current]:
            if cell_type[n] == WATER and n not in flood:
                flood.append(n)

    return cell_type


def getDistsToCoast(vertices, vert_regions, vert_neighbors, cell_type):
    dist_to_coast = [None for v in vertices]

    open_set = []
    for v in range(len(vertices)):
        ocean_count = 0
        land_count = 0
        for r in vert_regions[v]:
            if cell_type[r] == OCEAN:
                ocean_count += 1
            elif cell_type[r] == LAND:
                land_count += 1
        if ocean_count > 0:
            if land_count == 0:
                dist_to_coast[v] = 0
            else:
                dist_to_coast[v] = 0
                open_set.append(v)

    index = 0
    while index < len(open_set):
        cur = open_set[index]
        for v in vert_neighbors[cur]:
            cur_to_v = dist(vertices[cur], vertices[v])
            if dist_to_coast[v] is None or \
               dist_to_coast[v] > dist_to_coast[cur] + cur_to_v:
                dist_to_coast[v] = dist_to_coast[cur] + cur_to_v
                open_set.append(v)
        index += 1

    return dist_to_coast


def getTerrain(size, vertices, regions, cell_type, dist_to_coast, seed=None):
    if seed is None:
        seed = datetime.now().microsecond

    print("Calculating large-scale terrain...")

    print("Large-scale terrain seed: %d" % seed)
    np.random.seed(seed)

    noise = perlin.Perlin(size, 2, 3)

    elevations = [noise[v] for v in vertices]

    min_elevation = min(elevations)

    elevations = [(el - min_elevation)**2 * dc**0.2 if dc >= 0 else -1
            for el, dc in zip(elevations, dist_to_coast)]

    max_elevation = max(elevations)

    colors = [
            np.mean(np.array(elevations)[r]) / max(elevations) * (1-LAND) + LAND 
                if cell_type[i] == LAND
                else cell_type[i]
            for i, r in enumerate(regions)
            ]

    return colors



def main():
    size = 4096
    bbox = [(0,0),(0,size),(size,size),(size,0)]

    centers, neighbors, vertices, regions = getVoronoi(size, bbox)

    vert_neighbors = [[] for i in vertices]
    vert_regions = [[] for i in vertices]
    for r, p in enumerate(regions):
        for i, v in enumerate(p[:-1]):
            vert_regions[v].append(r)
            if p[i-1] not in vert_neighbors[v]:
                vert_neighbors[v].append(p[i-1])
            if p[i+1] not in vert_neighbors[v]:
                vert_neighbors[v].append(p[i+1])

    cell_type = getBasicShape(size, neighbors, vertices, regions, 32268)

    dist_to_coast = getDistsToCoast(vertices, vert_regions, 
            vert_neighbors, cell_type)


    patches = [Polygon(vertices[p], True, ec='k') for p in regions]

    patchCol = PatchCollection(patches, cmap=plt.get_cmap('terrain'))

    patchCol.set_clim(0, 1)
    patchCol.set_antialiased(False)

    fig, ax = plt.subplots()

    ax.add_collection(patchCol)

    ax.set_aspect('equal')
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    plt.tight_layout()
    plt.axis('off')

    for i in range(100):
        seed = datetime.now().microsecond

        colors = getTerrain(size, vertices, regions, cell_type, dist_to_coast, seed)
        
        patchCol.set_array(np.array(colors))

        plt.savefig('terrains/%06d.png' % seed, format='png', dpi=1200)
        #plt.show()

if __name__ == "__main__":
    main()

