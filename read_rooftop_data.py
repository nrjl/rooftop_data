import numpy as np
from scipy.io import loadmat
from imageio import imread
import os
import argparse
import matplotlib.pyplot as plt
import yaml
from data_downloader import MrDataGrabber

class RoofImage(object):
    def __init__(self, image_file):
        self.image_file = image_file
        self.mat_file = os.path.splitext(image_file)[0] + '.mat'  # Ensure associated mat file exists
        self.image_data = imread(image_file)
        try:
            mat_data = loadmat(self.mat_file)
        except FileNotFoundError:
            print('Associated mat file for {0} could not be loaded'.format(image_file))
            raise

        # roof_polygons is a list of polygons, each as an nx2 array of (x,y) points in pixel coords
        self.roof_polygons = np.array(mat_data['gt'][0])

    @staticmethod
    def _build_polygon_dict(points):
        polygon = []
        for p in points:
            polygon.append({'x': float(p[0]), 'y': float(p[1])})
        return {'points': polygon}

    def plot(self, ah=None):
        if ah is None:
            fh, ah = plt.subplots()
        ah.imshow(self.image_data)
        for p in self.roof_polygons:
            ah.plot(np.append(p[:,0], p[0,0]), np.append(p[:,1], p[0,1]), 'r-')

    def _get_outer_hull_dict(self):
        h, w, c = self.image_data.shape
        points = [[0, 0], [w, 0], [w, h], [0, h]]
        return self._build_polygon_dict(points)

    def _get_roof_dict(self):
        roofs = []
        for roof in self.roof_polygons:
            roofs.append(self._build_polygon_dict(roof))
        return roofs

    def make_yaml_message(self, target_name=None, target_dir=None):
        im_dir, im_file = os.path.split(self.image_file)
        # If neither dir nor name is specified, just put the yaml in the same place as the JPG
        if target_dir is None and target_name is None:
            target_dir = im_dir
        elif target_dir is None:
            target_dir = ''
        # If the target name is none, then add the yaml extension and put it in target_dir
        if target_name is None:
            target_name = os.path.splitext(im_file)[0] + '.yaml'

        full_dict = {'hull': self._get_outer_hull_dict()}
        full_dict['holes'] = self._get_roof_dict()

        with open(os.path.join(target_dir, target_name), 'wt') as fh:
            yaml.dump(full_dict, fh)


class RoofData(object):
    def __init__(self, data_location, data_dirs=['Rooftop/test', 'Rooftop/train']):
        self.data_location = data_location
        self.data_dirs = data_dirs
        self.roof_images = []

        for sd in data_dirs:                                                # Loop over test and train
            for image_file in os.listdir(os.path.join(data_location, sd)):       # Loop over files (finding jpgs)
                if image_file.endswith(".jpg") or image_file.endswith(".JPG"):
                    try:
                        self.roof_images.append(RoofImage(os.path.join(data_location, sd, image_file)))
                    except FileNotFoundError:
                        continue
        print('Loaded {0} images.'.format(len(self.roof_images)))

    def build_yamls(self, target_dir):
        # Create target dir if doesn't already exist
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for roof in self.roof_images:
            roof.make_yaml_message(target_dir=target_dir)
        print('{0} polygon roof definitions created in {1}.'.format(len(self.roof_images), target_dir))

    def print_stats(self):
        n_houses = np.array([roof.roof_polygons.shape[0] for roof in self.roof_images])
        print('{0} total roof polygons across {1} images.'.format(n_houses.sum(), len(n_houses)))
        print('Roofs per image: min {0}, max {1}, mean {2:0.2f}'.format(n_houses.min(), n_houses.max(), n_houses.mean()))


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
    all_roofs = RoofData(data_location=os.path.join(args.data_dir, 'Rooftop'))
    all_roofs.print_stats()
    all_roofs.build_yamls(target_dir=args.yaml_dir)
    if args.plot:
        n = np.random.randint(0, len(all_roofs.roof_images))
        print('Plotting image and {0} roof polygons for {1}'.format(len(all_roofs.roof_images[n].roof_polygons),
                                                                        all_roofs.roof_images[n].image_file))
        all_roofs.roof_images[n].plot()
        plt.show()