#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from datetime import datetime

import voronoi
import perlin

def dist(a, b):
    return np.linalg.norm(np.array(a) - b)

size = 4096
bbox = [(0,0),(0,size),(size,size),(size,0)]

print("Loading Voronoi cells...")

centers, neighbors, vertices, regions = np.load('voronoi32k.npy')

vert_neighbors = [[] for i in vertices]
vert_regions = [[] for i in vertices]
for r, p in enumerate(regions):
    for i, v in enumerate(p[:-1]):
        vert_regions[v].append(r)
        if p[i-1] not in vert_neighbors[v]:
            vert_neighbors[v].append(p[i-1])
        if p[i+1] not in vert_neighbors[v]:
            vert_neighbors[v].append(p[i+1])

patches = [Polygon(vertices[p], True, ec='k') for p in regions]

patchCol = PatchCollection(patches, cmap=plt.get_cmap('terrain'))

print("Calculating landscape...")

seed = datetime.now().microsecond
print("Landmass shape seed: %d" % seed)
np.random.seed(seed)

noise = perlin.Perlin(size, 8) #np.load('noise.npy')

LAND = 0.25
WATER = 0.15
OCEAN = 0

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

patchCol.set_array(np.array(cell_type))
patchCol.set_clim(0, 1)
patchCol.set_antialiased(False)

fig, ax = plt.subplots()

ax.add_collection(patchCol)

ax.set_aspect('equal')
ax.set_xlim(0, size)
ax.set_ylim(0, size)
plt.tight_layout()
plt.axis('off')

plt.figtext(0.05, 0.05, '%d' % seed)

plt.savefig('shape.png', format='png', dpi=1200)
plt.show()

# for p in regions:
#     plt.plot(*vertices[p].T, c='k', linewidth=1)
# 
# for i, ns in enumerate(neighbors):
#     plt.scatter(*centers[i].T, c='b')
#     for j in ns:
#         if j > i:
#             plt.plot(*centers[[i,j]].T, c='b')
# 
# plt.show()
