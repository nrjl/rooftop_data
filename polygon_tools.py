import numpy as np
import matplotlib.pyplot as plt
import yaml


class MaxIterationsException(Exception):
    pass

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


def on_segment(segment, point):
    # Check if point is on segment (assuming they are colinear)
    if np.all(point > segment.min(axis=0)) and np.all(point < segment.max(axis=0)):
        return True
    return False

def orientation(p0, p1, p2):
    # Orientation for traversing around 3 points (p0->p1->p2)
    # Return is -1 for counterclockwise, 0 for colinear, 1 for clockwise
    d10 = p1-p0
    d21 = p2-p1
    orient = d10[1]*d21[0] - d21[1]*d10[0]
    return np.sign(orient)

def segment_intersection(seg0, seg1):
    # Check if two segments intersect
    # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    o1 = orientation(seg0[0], seg0[1], seg1[0])
    o2 = orientation(seg0[0], seg0[1], seg1[1])
    o3 = orientation(seg1[0], seg1[1], seg0[0])
    o4 = orientation(seg1[0], seg1[1], seg0[1])

    if (o1 != o2) and (o3!=o4):
        return True

    # Special Cases
    # seg1[0] lies on seg0 (colinear and on segment)
    if (o1 == 0) and on_segment(seg0, seg1[0]):
        return True

    # seg1[1] lies on seg0
    if (o2 == 0) and on_segment(seg0, seg1[1]):
        return True

    # seg0[0] lies on seg1
    if (o3 == 0) and on_segment(seg1, seg0[0]):
        return True

    # seg0[1] lies on seg1
    if (o4 == 0) and on_segment(seg0, seg0[1]):
        return True

    return False


class Polygon(np.ndarray):
    # Basically subclass the numpy ndarray object, slightly cheatily, so I can add the point in polygon methods
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        assert len(obj.shape) is 2, "Polygon must be created with 2D (nx2) input array"
        assert obj.shape[1] is 2, "Polygon must be created with nx2 input array"
        return obj

    def _outside_limits(self, p):
        # Return true if point is definitely outside (boundary check)
        if np.any(p < self.min(axis=0)) or np.any(p > self.max(axis=0)):
            return True
        else:
            return False

    def point_inside_cn(self, p):
        #  crossing_number(): crossing number test for a point in a polygon
        #       Input:   P = a point,
        #                V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
        #       Return:  False = outside, True = inside
        #  This code is patterned after [Franklin, 2000]
        if self._outside_limits(p):
            return False

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

    def point_inside_wn(self, p):
        #  winding_number(): winding number test for a point in a polygon
        #       Input:   p = a point,
        #                V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
        #       Return:  False = outside, True = inside
        # wn = the winding number (=0 only when P is outside)
        if self._outside_limits(p):
            return False

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

    def point_inside(self, p):
        # Default to winding number test
        return self.point_inside_wn(p)

    def origin_offset(self):
        return self - self[0]

    def rotate(self, angle, units='rad'):
        if angle == 'deg':
            angle *= np.pi/180
        rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        return np.matmul(self, rot_matrix)

    def place_inside(self, obs, max_attempts = 1e2):
        # Randomly rotate and shift the obstacle so that it fits inside the hull
        # NOTE: Currently assumes hull is rectangular!
        angle = np.random.uniform(-np.pi, np.pi)
        new_obs = obs.rotate(angle)
        in_hull = False
        n_attempts = 0
        while not in_hull:
            n_attempts += 1
            if n_attempts > max_attempts:
                raise MaxIterationsException('Maximum attempts to place obs in hull reached!')
            shift = np.random.uniform(self.min(axis=0)-new_obs.min(axis=0), self.max(axis=0)-new_obs.max(axis=0))
            in_hull = True
            for p in new_obs:
                if not self.point_inside_wn(p + shift):
                    in_hull = False
                    break
        if not in_hull:
            raise MaxIterationsException
        return new_obs + shift

    def overlap(self, other_poly):
        # First check if they are outside each other's bounding boxes
        if np.any(other_poly.max(axis=0) < self.min(axis=0)) or np.any(other_poly.min(axis=0) > self.max(axis=0)):
            return False

        # Next check if any points are contained in other:
        for p in other_poly:
            if self.point_inside(p):
                return True
        for p in self:
            if other_poly.point_inside(p):
                return True

        # Finally, check for intersecting edges
        for i in range(len(self)):
            ip = (i+1)%len(self)
            for j in range(len(other_poly)):
                jp = (j+1)%len(other_poly)
                if segment_intersection(self[[i,ip]], other_poly[[j,jp]]):
                    return True

        return False

    def plot(self, ah, ls='r-'):
        return ah.plot(np.append(self[:,0], self[0,0]), np.append(self[:,1], self[0,1]), ls)

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
    def __init__(self, hull, obstacles, scale=1.0):
        self.hull = Polygon(hull)                             # outer hull as a polygon
        self.obstacles = [Polygon(obs) for obs in obstacles]  # list of (Polygon) obstacles
        self.scale = scale

    @staticmethod
    def _build_polygon_dict(points, scale=1.0):
        polygon = []
        for p in points:
            polygon.append({'x': scale * float(p[0]), 'y': scale * float(p[1])})
        return {'points': polygon}

    def _plot_obstacles(self, ah, *args, **kwargs):
        out = []
        for p in self.obstacles:
            out.extend(p.plot(ah, *args, **kwargs))
        return out

    def plot(self, ah=None):
        if ah is None:
            fh, ah = plt.subplots()
        out = self.hull.plot(ah, 'b-')
        out.extend(self._plot_obstacles(ah, 'r-'))
        return out

    def _build_obstacle_list(self):
        obstacle_list = []
        for obs in self.obstacles:
            obstacle_list.append(self._build_polygon_dict(obs, self.scale))
        return obstacle_list

    def make_yaml_message(self, filename):
        full_dict = {'hull': self._build_polygon_dict(self.hull, self.scale)}
        full_dict['holes'] = self._build_obstacle_list()

        with open(filename, 'wt') as fh:
            yaml.dump(full_dict, fh)
