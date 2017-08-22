#!/usr/bin/python

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

seed = datetime.now().microsecond
np.random.seed(seed)

size = 4096
bbox = [(0,0),(0,size),(size,size),(size,0)]

print "Loading Voronoi cells..."

centers, neighbors, vertices, regions = np.load('voronoi32k.npy')
# centers, neighbors, vertices, regions = \
#         voronoi.relaxed_voronoi(32000, bbox)
# 
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

patches = [Polygon(vertices[p], True, ec='k') for p in regions]

patchCol = PatchCollection(patches, cmap=plt.get_cmap('terrain'))

print "Calculating landscape..."

noise = np.load('noise.npy')

LAND = 0.25
WATER = 0.15
OCEAN = 0

cell_type = []

def isLand(c):
    v = [int(t) for t in c]

    x = 2. * c[0] / size - 1
    y = 2. * c[1] / size - 1

    return noise[v[0]][v[1]] > 0.316 * (1 + x**2 + y**2)

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
            dist_to_coast[v] = -1
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

peaks = [(1300, 2500), (1500, 3100), (1950, 3200)]
dist_to_peak = [None for d in dist_to_coast]
for i, v in enumerate(vertices):
    dist_to_peak[i] = 1/np.mean([1/dist(v, p) for p in peaks])

print "Calculating elevations..."

seed = 17770 #datetime.now().microsecond
np.random.seed(seed)
noise1 = perlin.Perlin(size, 6, 3)

seed = datetime.now().microsecond
print "Large-scale elevation seed: %d" % seed
np.random.seed(seed)
noise2 = perlin.Perlin(size, 1, 0)

m = 1 - mpimg.imread('noisy_mountains.png').mean(axis=-1)

mdtc = max(dist_to_coast)
mdtp = max(dist_to_peak)

elevations = [
        2 * noise1.at(*v) + 
        .4 * noise2.at(*v) + 
        m[int(size-v[1]-1)][int(v[0]-1)]
        #(dc / (dp+dc))**2 if dc <= dp else (1/dp)**.35
        for v in vertices
        ]

min_el = min(elevations)

elevations = [
        (el-min_el)**1 * dc**0.2 if dc >= 0 else -1
        for el, dc in zip(elevations, dist_to_coast)
        ]

lvs = 10.
max_el = float(max(elevations))
els = [
        OCEAN if x < 0 else x/max_el
        for i, x in enumerate(elevations)
      ]

clrs = [
        int(np.mean(np.array(els)[r])*lvs)/lvs * .75 + .25
            if cell_type[i] == LAND 
            else cell_type[i]
        for i, r in enumerate(regions)
       ]

patchCol.set_array(np.array(clrs))
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
