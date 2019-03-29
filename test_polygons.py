import numpy as np
import matplotlib.pyplot as plt
from polygon_tools import Polygon, PolygonScene

point1 = np.array([0.5, 0.5])
point2 = np.array([1, 1])
poly1 = Polygon([[0,0], [1,0], [0.5, 1]])
poly2 = Polygon([[0.2,0.1], [0.8,0.1], [0.5, 0.7]])
poly3 = Polygon([[0.5,0.8], [1,1.8], [0, 1.8]])
poly4 = poly3 - np.array([0, 0.25])
hull = Polygon([[0,0], [1.5,0], [1.5, 2], [0,2]])
scene = PolygonScene(hull, [poly1, poly2, poly3, poly4])

assert poly1.overlap(poly2)
assert poly2.overlap(poly1)
assert poly3.overlap(poly1)
assert poly3.overlap(poly4)
assert hull.overlap(poly2)
assert hull.overlap(poly1)
assert not poly3.overlap(poly2)
assert poly1.point_inside(point1)
assert not poly1.point_inside(point2)

fh, ah = plt.subplots()

h1 = ah.plot(point1[0], point1[1], 'r.')
h2 = ah.plot(point2[0], point2[1], 'b.')
hp = []
for p, l in zip(scene.obstacles, ['g-', 'y-', 'k-', 'm-']):
    hp.extend(p.plot(ah, l))
hp.extend(hull.plot(ah, 'b-'))
# hs = scene.plot(ah)
ah.legend(hp, ['poly1', 'poly2', 'poly3', 'poly4', 'hull'])
plt.show(block=False)
