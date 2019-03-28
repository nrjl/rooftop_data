import numpy as np
import matplotlib.pyplot as plt
from polygon_tools import PolygonScene, MaxIterationsException, Polygon
import os
import argparse
from read_rooftop_data import RoofDataset, MrDataGrabber, safe_mkdir

def create_synthetic_data(hull, obstacle_list, n_obstacles, n_samples, target_dir='./', max_failures=100):
    # Use same outer hull for all

    for i in range(n_samples):
        # Add obstacles by sampling from all roofs
        obs = []
        n_failures = 0
        while len(obs) < n_obstacles and n_failures < max_failures:
            try:
                new_obs = hull.place_inside(np.random.choice(obstacle_list))
            except MaxIterationsException:
                print('Could not place obstacle in hull in max iterations. Continuing.')
                n_failures += 1
                continue
            obs.append(new_obs)
        if n_failures >= max_failures:
            raise MaxIterationsException('Too many failed attempts at placing obstacles in hull!')
        new_scene = PolygonScene(hull, obs)
        outfile = os.path.join(target_dir, '{0:04d}.yaml'.format(i))
        new_scene.make_yaml_message(outfile)
        print('Generated new sample {0}, with {1} obstacles.'.format(outfile, len(obs)))
    return new_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a set of random maps using roof polygons from EPFL dataset')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot a random image from the database')
    parser.add_argument('-y', '--yaml_dir', default='./data/random', help='Target location for yaml files')
    parser.add_argument('-d', '--data_dir', default='./data', help='Target location for downloading data')
    parser.add_argument('-ns', '--n_samples', default=1, help='Number of random samples')
    parser.add_argument('-no', '--n_obstacles', default=5, help='Number of obstacles (roof polygons) per sample')
    parser.add_argument('--hull_size', nargs=2, default=[-1, -1], help='Specific hull size w h (default to dataset image size)')
    args = parser.parse_args()

    safe_mkdir(args.data_dir)
    safe_mkdir(args.yaml_dir)
    data_grabber = MrDataGrabber('http://cvlab.epfl.ch/wp-content/uploads/2018/08/Rooftop.zip', args.data_dir)
    data_grabber.download()
    epfl_roofscenes = RoofDataset(data_location=os.path.join(args.data_dir, 'Rooftop'))
    epfl_roofscenes.print_stats()

    if args.hull_size[0] < 0:
        hull = epfl_roofscenes.roof_scenes[0].hull
    else:
        hull = Polygon([[0, 0], [args.hull_size[0], 0], [args.hull_size[0], args.hull_size[1]], [0, args.hull_size[1]]])
    final_scene = create_synthetic_data(hull, epfl_roofscenes.all_roofs, n_obstacles=args.n_obstacles,
                                        n_samples=args.n_samples, target_dir=args.data_dir)
    if args.plot:
        final_scene.plot()
        plt.show(block=False)