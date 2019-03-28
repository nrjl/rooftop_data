import numpy as np
from scipy.io import loadmat
from imageio import imread
import os
import argparse
import matplotlib.pyplot as plt
from data_downloader import MrDataGrabber
from polygon_tools import Polygon, PolygonScene


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
        self.hull = Polygon([[0, 0], [w, 0], [w, h], [0, h]])
        self.obstacles =  [Polygon(obs) for obs in np.array(mat_data['gt'][0])]

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
        # Use same outer hull for all
        if hull is None:
            hull = self.roof_scenes[0].hull

        for i in range(n_samples):
            # Add obstacles by sampling from all roofs
            pass

    def print_stats(self):
        n_roofs = np.array([len(scene.obstacles) for scene in self.roof_scenes])
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