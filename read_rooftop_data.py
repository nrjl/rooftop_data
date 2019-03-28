import numpy as np
from scipy.io import loadmat
from imageio import imread
import os
import argparse
import matplotlib.pyplot as plt
import yaml
from data_downloader import MrDataGrabber

# NOTE: All polygons are assumed to be automatically closed, so the polygon definition contains each vertex once only,
# and the last vertex is assumed to connect to the first vertex (i.e. first vertex should not be repeated at the end)

class PolygonScene(object):
    def __init__(self, hull, obstacles):
        # hull describes the outer hull as a polygon (assume closed between first and last point)
        self.hull = hull

        # obstacles is a list of obstacles, each as an nx2 array of (x,y) points in pixel coords
        self.obstacles = obstacles


    @staticmethod
    def _build_polygon_dict(points):
        polygon = []
        for p in points:
            polygon.append({'x': float(p[0]), 'y': float(p[1])})
        return {'points': polygon}

    def _plot_poly(self, ah, p, ls='r-'):
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



class RoofScene(PolygonScene):
    # Inherits from normal RoofPolygon, but in this case also has an image, from which the outer hull is built

    def __init__(self, image_file):
        self.image_file = image_file
        self.mat_file = os.path.splitext(image_file)[0] + '.mat'  # Ensure associated mat file exists
        self.image_data = imread(image_file)
        try:
            mat_data = loadmat(self.mat_file)
        except FileNotFoundError:
            print('Associated mat file for {0} could not be loaded'.format(image_file))
            raise

        h, w, c = self.image_data.shape
        self.hull = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        self.obstacles = np.array(mat_data['gt'][0])

    def plot(self, ah=None):
        if ah is None:
            fh, ah = plt.subplots()
        ah.imshow(self.image_data)
        super(RoofScene, self).plot(ah=ah)

    def make_yaml_message(self, target_file=None, target_dir=None):
        im_dir, im_file = os.path.split(self.image_file)
        # If neither dir nor name is specified, just put the yaml in the same place as the JPG
        if target_dir is None and target_file is None:
            target_dir = im_dir
        elif target_dir is None:
            target_dir = ''
        # If the target name is none, then add the yaml extension and put it in target_dir
        if target_file is None:
            target_file = os.path.splitext(im_file)[0] + '.yaml'

        target_file = os.path.join(target_dir, target_file)
        return super(RoofScene, self).make_yaml_message(target_file)


class RoofDataset(object):
    def __init__(self, data_location, data_dirs=['Rooftop/test', 'Rooftop/train']):
        self.data_location = data_location
        self.data_dirs = data_dirs
        self.roof_scenes = []           # List of individual scenes from dataset
        self.all_roofs = []             # List of all the roofs from the dataset (for generating synthetic data)

        for sd in data_dirs:                                                # Loop over test and train
            for image_file in os.listdir(os.path.join(data_location, sd)):       # Loop over files (finding jpgs)
                if image_file.endswith(".jpg") or image_file.endswith(".JPG"):
                    try:
                        new_scene = RoofScene(os.path.join(data_location, sd, image_file))
                        self.roof_scenes.append(new_scene)
                        self.all_roofs.extend(new_scene.obstacles)
                    except FileNotFoundError:
                        continue
        print('Loaded {0} images.'.format(len(self.roof_scenes)))

    def build_yamls(self, target_dir):
        # Create target dir if doesn't already exist
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for scene in self.roof_scenes:
            scene.make_yaml_message(target_dir=target_dir)
        print('{0} polygon roof definitions created in {1}.'.format(len(self.roof_scenes), target_dir))

    def create_synthetic_data(self, n_obstacles, n_samples, hull=None):
        if hull is None:
            hull = self.roof_scenes[0].hull

        for i in range(n_samples):
            pass


    def print_stats(self):
        n_roofs = np.array([scene.obstacles.shape[0] for scene in self.roof_scenes])
        print('{0} total roof polygons across {1} images.'.format(n_roofs.sum(), len(n_roofs)))
        print('Roofs per image: min {0}, max {1}, mean {2:0.2f}'.format(n_roofs.min(), n_roofs.max(), n_roofs.mean()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to remove bad data from a database')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot a random image from the database')
    parser.add_argument('-y', '--yaml_dir', default='./data/Rooftop/yaml', help='Target location for yaml files')
    parser.add_argument('-d', '--data_dir', default='./data', help='Target location for downloading data')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    data_grabber = MrDataGrabber('http://cvlab.epfl.ch/wp-content/uploads/2018/08/Rooftop.zip', args.data_dir)
    data_grabber.download()
    epfl_roofscenes = RoofDataset(data_location=os.path.join(args.data_dir, 'Rooftop'))
    epfl_roofscenes.print_stats()
    epfl_roofscenes.build_yamls(target_dir=args.yaml_dir)
    if args.plot:
        rand_scene = np.random.choice(epfl_roofscenes.roof_scenes)
        print('Plotting image and {0} roof polygons for {1}'.format(len(rand_scene.obstacles), rand_scene.image_file))
        rand_scene.plot()
        plt.show(block=False)