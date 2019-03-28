import numpy as np
import matplotlib.pyplot as plt
import yaml

# NOTE: All Polygons are assumed to be automatically closed, so the polygon definition contains each vertex ONCE only,
# and the last vertex is assumed to connect to the first vertex (i.e. first vertex should not be repeated at the end)

# ===================================================================
# Adapted from C++ code by Dan Sunday:
#  Copyright 2000 softSurfer, 2012 Dan Sunday
#  This code may be freely used and modified for any purpose
#  providing that this copyright notice is included with it.
#  SoftSurfer makes no warranty for this code, and cannot be held
#  liable for any real or imagined damage resulting from its use.
#  Users of this code must verify correctness for their application.
# http://geomalgorithms.com/a03-_inclusion.html

def is_left(p0, p1, p2):
    #  is_left(): tests if a point is Left|On|Right of an infinite line.
    #     Input:  three points P0, P1, and P2
    #     Return: >0 for P2 left of the line through P0 and P1
    #             =0 for P2  on the line
    #             <0 for P2  right of the line
    #     See: Algorithm 1 "Area of Triangles and Polygons"
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


class Polygon(np.ndarray):
    # Basically subclass the numpy ndarray object, slightly cheatily, so I can add the point in polygon methods
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        assert len(obj.shape) is 2, "Polygon must be created with 2D (nx2) input array"
        assert obj.shape[1] is 2, "Polygon must be created with nx2 input array"
        return obj

    def crossing_number(self, p):
        #  crossing_number(): crossing number test for a point in a polygon
        #       Input:   P = a point,
        #                V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
        #       Return:  0 = outside, 1 = inside
        #  This code is patterned after [Franklin, 2000]
        cn = 0                                                  # crossing number counter
        #  loop through all edges of the polygon
        for i in range(self.shape[0]):                          # edge from V[i]  to V[i+1]
            ip = (i + 1) % self.shape[0]
            if (self[i, 1] <= p[1] and self[ip, 1] > p[1]) or (self[i, 1] > p[1] and self[ip, 1] <= p[1]):   # a downward crossing
                # compute the actual edge-ray intersect x-coordinate
                vt = (p[1] - self[i, 1]) / (self[ip, 1] - self[i, 1])
                if p[0] < (self[i, 0] + vt * (self[ip, 0] - self[i, 0])):  #  P.x < intersect
                     cn += 1                                    # a valid crossing of y=P.y right of P.x
        return bool(cn % 2)                                     # 0 if even (out), and 1 if  odd (in)

    def winding_number(self, p):
        #  winding_number(): winding number test for a point in a polygon
        #       Input:   p = a point,
        #                V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
        #       Return:  wn = the winding number (=0 only when P is outside)
        wn = 0    #  the  winding number counter

        #  loop through all edges of the polygon
        for i in range(self.shape[0]):                      # edge from V[i] to  V[i+1]
            ip = (i + 1) % self.shape[0]                    # This basically just wraps back to the start point
            if self[i, 1] <= p[1]:                          # start y <= P.y
                if self[ip, 1]  > p[1]:                     # an upward crossing
                     if is_left(self[i], self[ip], p) > 0:  # P left of  edge
                         wn += 1                            # have a valid up intersect

            else:                                           # start y > P.y (no test needed)
                if self[ip, 1] <= p[1]:                     # a downward crossing
                    if is_left(self[i], self[ip], p) < 0:   # P right of  edge
                        wn -= 1                             # have  a valid down intersect
        return bool(wn)

    def origin_offset(self):
        return self - self[0]

    def rotate(self, angle, units='rad'):
        if angle == 'deg':
            angle *= np.pi/180
        R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        return np.matmul(self, R)

# ===================================================================

def plot_poly_test(V,n):
    fig1, ax1 = plt.subplots(1, 1)
    PX = np.random.uniform(V[:,0].min()-1, V[:,0].max()+1, (n,1))
    PY = np.random.uniform(V[:,1].min()-1, V[:,1].max()+1, (n,1))
    ax1.plot(V[:,0], V[:,1], 'b-')
    for i in range(n):
        if V.winding_number([PX[i],PY[i]]):
            ax1.plot(PX[i],PY[i], 'g.')
        else:
            ax1.plot(PX[i],PY[i], 'r.')


class PolygonScene(object):
    def __init__(self, hull, obstacles):
        self.hull = Polygon(hull)                             # outer hull as a polygon
        self.obstacles = [Polygon(obs) for obs in obstacles]  # list of (Polygon) obstacles

    @staticmethod
    def _build_polygon_dict(points):
        polygon = []
        for p in points:
            polygon.append({'x': float(p[0]), 'y': float(p[1])})
        return {'points': polygon}

    @staticmethod
    def _plot_poly(ah, p, ls='r-'):
        ah.plot(np.append(p[:,0], p[0,0]), np.append(p[:,1], p[0,1]), ls)

    def _plot_obstacles(self, ah, *args, **kwargs):
        for p in self.obstacles:
            self._plot_poly(ah, p, *args, **kwargs)

    def plot(self, ah=None):
        if ah is None:
            fh, ah = plt.subplots()
        self._plot_poly(ah, self.hull, 'b-')
        self._plot_obstacles(ah, 'r-')

    def _build_obstacle_list(self):
        obstacle_list = []
        for obs in self.obstacles:
            obstacle_list.append(self._build_polygon_dict(obs))
        return obstacle_list

    def make_yaml_message(self, filename):
        full_dict = {'hull': self._build_polygon_dict(self.hull)}
        full_dict['holes'] = self._build_obstacle_list()

        with open(filename, 'wt') as fh:
            yaml.dump(full_dict, fh)