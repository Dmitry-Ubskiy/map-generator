#!/usr/bin/python3

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon

import progressbar

def intersection(p1, p2):
    p1 = Polygon(p1)
    p2 = Polygon(p2)

    return np.array([p for p in p1.intersection(p2).exterior.coords])

def region_corners(vor, bbox):
    radius = np.ptp(bbox, axis=0) * 2

    center = np.mean(bbox, axis=0)

    all_ridges = {}
    for (p1,p2),(v1,v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2,v1,v2))
        all_ridges.setdefault(p2, []).append((p1,v1,v2))

    regions = []
    for index, point in enumerate(vor.point_region):
        ridges = all_ridges[index]
        region = vor.regions[point]
        corners = [vor.vertices[v] for v in region if v >= 0]

        for p, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p] - vor.points[index]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            mid = vor.points[[index, p]].mean(axis=0)
            direction = np.sign(np.dot(mid - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            corners.append(far_point)

        corners = np.array(corners)
        c = corners.mean(axis=0)
        angles = np.arctan2(corners[:,1]-c[1], corners[:,0]-c[0])
        corners = corners[np.argsort(angles)]

        regions.append(intersection(corners, bbox))

    return np.array(regions)

def centroid(p):
    p = list(p)
    if not list(p[0]) == list(p[-1]):
        p.append(p[0])
    n = len(p) - 1

    t = [
        p[i][0] * p[i+1][1] - p[i+1][0] * p[i][1] 
        for i in range(n)
        ]

    A = 3. * sum(t)

    x = sum([
        (p[i][0] + p[i+1][0]) * t[i]
        for i in range(n)
        ]) / A

    y = sum([
        (p[i][1] + p[i+1][1]) * t[i]
        for i in range(n)
        ]) / A

    return [x, y]


def share_side(p1, p2):
    cnt = 0
    for i in p1[:-1]:
        for j in p2[:-1]:
            if np.all(i == j):
                cnt += 1
    return cnt


def relaxed_voronoi(N, bbox, n_iter=2):
    x_min = min([p[0] for p in bbox])
    x_max = max([p[0] for p in bbox])
    y_min = min([p[1] for p in bbox])
    y_max = max([p[1] for p in bbox])
    xs = np.random.uniform(x_min, x_max, N)
    ys = np.random.uniform(y_min, y_max, N)
    points = np.stack((xs, ys)).T
    vor = Voronoi(points)

    for i in range(n_iter):
        c = []
        p = region_corners(vor, bbox)
        for i in range(N):
            c.append(centroid(p[i]))
        vor = Voronoi(np.array(c))

    centers = vor.points
    order = np.argsort(np.linalg.norm(vor.points, axis=1))

    centers = centers[order]
    polygons = region_corners(vor, bbox)[order]

    radii = []
    for i, p in enumerate(polygons):
        radii.append(max(np.linalg.norm(centers[i] - p, axis=1)))

    neighbors = [[] for i in range(N)]

    bar = progressbar.ProgressBar(max_value=len(polygons))

    for i, p1 in enumerate(polygons):
        for j, p2 in enumerate(polygons):
            if i >= j:
                continue
            if np.linalg.norm(centers[i] - centers[j]) > \
               2*(radii[i] + radii[j]):
                continue
            if share_side(p1, p2) >= 2:
                neighbors[i].append(j)
                neighbors[j].append(i)
        bar.update(i)
    bar.finish()

    vertices = []
    new_polygons = []
    for p in polygons:
        new_polygons.append([])
        for v in p:
            if list(v) not in vertices:
                vertices.append(list(v))
            new_polygons[-1].append(vertices.index(list(v)))

    vertices = np.array(vertices)
    new_polygons = np.array(new_polygons)

    return (centers, neighbors, vertices, new_polygons)

