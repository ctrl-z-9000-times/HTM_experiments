"""
Dataset tools for labeled image files.  

Dataset format is a directory containing:
    Image files,
    Label files,
    labels.csv

Image files:
    Drag and drop image files into the dataset directory before running the
    label tool.  The dataset directory can contain subdirectories, which are
    scanned for images as well.  Acceptable image formats: anything PIL/Pillow
    can open.

Label files:
    Label files are the output of the GUI tool.  They are RGBA images where the 
    color represents the label.  The correspondance between colors and labels is
    stored in labels.csv.  Label files are named after the image file they are 
    paired with, for example:  MyImage.bmp has label file MyImage.label.png

labels.csv:
    labels.csv is a comma seperated values file matching colors with labels.
    The entry 'unlabeled' is always present as the color black.
    It has columns "R", "G", "B", "A" and "label"
"""

import numpy as np
import os
from send2trash import send2trash
import random
import csv
from PIL import Image, ImageDraw
import scipy
from copy import deepcopy

def read_names(path):
    """
    Returns dict of {label-color: label-name}

    Returns the contents of the labels.csv file for the given database.
    Returns an empty dictionary if no labels.csv file is found.
    """
    labels = {}
    try:
        with open(os.path.join(path, 'labels.csv'), 'r') as label_file:
            label_reader = csv.DictReader(label_file)
            for entry in label_reader:
                color = (entry['R'], entry['G'], entry['B'], entry['A'])
                color = tuple(int(cc) for cc in color)
                # try:
                #     # Greyscale color, cast to integer
                #     color = int(color)
                # except ValueError:
                #     # Assume color is a tuple or list of integers
                #     color_components = color.strip('() ').split(',')
                #     color_components = [cc.strip() for cc in color_components]
                #     color = tuple(int(cc) for cc in color_components)
                label = entry['label']
                labels[color] = label
    except FileNotFoundError:
        pass
    return labels

def find_dataset(path):
    """
    Returns pair of (images, labels)
        Where images is list of filepaths,
        Where labels is list of filepaths, or None if the corresponding image
              has no labels file,
    """
    image_extensions = [
        '.bmp',
        '.dib',
        '.png',
        '.jpg',
        '.jpeg',
        '.jpe',
        '.tif',
        '.tiff',
    ]
    images = []
    labels = []
    for dirpath, dirnames, filenames in os.walk(path):
        for fn in filenames:
            basename, ext = os.path.splitext(fn)
            if ext.lower() not in image_extensions:
                continue    # This file is not an image, ignore it.
            file_path = os.path.join(dirpath, fn)
            if basename.endswith(".label"):
                labels.append(file_path)
            else:
                images.append(file_path)

    # Match the images with their labels.
    images.sort()
    labels.sort()
    # Insert None's into the labels list where there are missing labels.
    for index, image in enumerate(images):
        image_name, image_ext = os.path.splitext(image)
        try:
            label = labels[index]   # This should be the corresponding label file
            label_name, label_ext       = os.path.splitext(label)
            image_from_label, dot_label = os.path.splitext(label_name)
            has_label = (image_name == image_from_label)     # File names match
        except IndexError:  # index >= len(labels)
            has_label = False
        if not has_label:
            labels.insert(index, None)
    return images, labels

class Dataset:
    """
    This is the database backend for the label tool GUI.
    This manages the data and datasets for some experiments.

    Attribute path      Path to base directory containing
    Attribute names     Dict of {color: label-name}
    Attribute images    List of paths to the image data files
    Attribute labels    List of paths to the image label files
    Attribute cursor    ...
    Attribute sorted_names  Just the names (strings) in sorted order.
                            This is used to convert labels to integer indecies for
                            where colors (random 32 bit integers) are unacceptable.
    Attribute unlabeled_index   Index of 'unlabeled' in self.sorted_names
                                (or None if not present)

    """
    def __init__(self, path=None):
        if path is not None:
            self.load_dataset(path)
        else:
            self.path   = None
            self.images = []
            self.labels = []
            self.names  = {}
            self.sorted_names = []
            self.cursor = None

    def load_dataset(self, path):
        """Loads a new database in, discards the old one"""
        self.path   = path
        self.images, self.labels = find_dataset(path)
        self.names  = read_names(path)
        self.sorted_names = sorted(self.names.values())
        try:
            self.unlabeled_index = self.sorted_names.index('unlabeled')
        except ValueError:
            self.unlabeled_index = None
        self.cursor = 0

    @property
    def current_image(self):
        return self.images[self.cursor]

    @property
    def current_label(self):
        lbl = self.labels[self.cursor]
        if lbl is None:
            size = Image.open(self.current_image).size
            img  = Image.new('RGBA', size)
            img.putalpha(0)
            data_name, data_ext = os.path.splitext(self.current_image)
            lbl  = data_name + ".label.png"
            img.save(lbl)
            self.labels[self.cursor] = lbl
        return lbl

    def next_image(self):
        self.cursor += 1
        if self.cursor >= len(self.images):
            self.cursor = 0

    def prev_image(self):
        self.cursor -= 1
        if self.cursor < 0:
            self.cursor = len(self.images) - 1

    def random_image(self):
        self.cursor = random.randrange(0, len(self.images))

    def __len__(self):
        """Returns the number of images in the currently laoded dataset."""
        return len(self.images)

    def delete_current_image(self):
        send2trash(self.current_image)
        send2trash(self.current_label)
        self.images.pop(self.cursor)
        self.labels.pop(self.cursor)
        if self.cursor >= len(self.images):
            self.cursor = 0

    def add_label_outline(self, label, outline):
        """
        Save the given label to file.

        Argument label can be a color or a string
        Argument outline is list of pairs of (x, y) coordinates of a polygon
                 If the length of outline is less than 2, this does nothing.
        """
        if isinstance(label, str):
            label = next(c for c, nm in self.names.items() if nm == label)
        assert(isinstance(label, tuple) and len(label) == 4) # Colors are tuples of 4 ints
        if len(outline) < 2:
            return # Already done.
        im = Image.open(self.current_label)
        # Draw the polygon
        draw = ImageDraw.Draw(im)
        draw.polygon(outline, fill=label)
        del draw
        im.save(self.current_label)

    def add_label_mask(self, label, mask):
        """
        Save the given label to file.

        Argument label can be a color or a string
        Argument boolean image, True where label will be set.
        """
        if isinstance(label, str):
            label = next(c for c, nm in self.names.items() if nm == label)
        assert(isinstance(label, tuple) and len(label) == 4) # Colors are tuples of 4 ints
        assert(mask.dtype == np.bool)
        img = Image.open(self.current_label)
        data = np.array(img, dtype=np.uint8)
        data[mask] = label
        size = tuple(reversed(data.shape[:2]))
        new_img = Image.frombuffer("RGBA", size, data, "raw", "RGBA", 0, 1)
        new_img.save(self.current_label)

    def get_unused_color(self):
        used_colors   = set(self.names.keys())
        new_color     = None
        while not new_color or new_color in used_colors:
            # Generate a random color
            new_color = tuple(random.randrange(0, 2**32).to_bytes(4, 'little'))
        return new_color

    def add_label_type(self, label):
        """
        Adds a new entry to a labels.csv file
        """
        color = self.get_unused_color()
        # Check that the new entry is valid
        assert(color not in self.names.keys())
        assert(label not in self.names.values())
        # Open and ready the label names file.
        with open(os.path.join(self.path, 'labels.csv'), 'a') as label_file:
            label_fieldnames = ['R', 'G', 'B', 'A', 'label']
            label_writer = csv.DictWriter(label_file, fieldnames=label_fieldnames)
            if not self.names:      # labels.csv file not found.
                label_writer.writeheader()
                label_writer.writerow({
                    'R': 0,
                    'G': 0,
                    'B': 0,
                    'A': 0,
                    'label': 'unlabeled',
                })
            # Append the new entry
            label_writer.writerow({
                'R': color[0],
                'G': color[1],
                'B': color[2],
                'A': color[3],
                'label': label
            })
        # Update the names with the new entry
        self.names = read_names(self.path)

    def discard_unlabeled_data(self):
        """
        Removes from the current image pool all images which are either missing
        a labels image of whos labels image contains no labels.
        """
        self.images = [im for idx, im in enumerate(self.images) if self.labels[idx] is not None]
        self.labels = [lbl for lbl in self.labels if lbl is not None]
        for idx in range(len(self.labels)-1, -1, -1):
            # Load each label file and discard the ones which are all zero
            lbl = np.asarray(Image.open(self.labels[idx]))
            if np.all(lbl == 0):
                self.images.pop(idx)
                self.labels.pop(idx)

    def discard_labeled_data(self):
        """
        Removes from the current image pool all images which have labels.
        """
        for idx in range(len(self.labels)-1, -1, -1):
            # No label file exists, ok.
            if self.labels[idx] is None:
                continue
            # Load each label file and discard the ones which are not all zero.
            lbl = np.asarray(Image.open(self.labels[idx]))
            if not np.all(lbl == 0):
                self.images.pop(idx)
                self.labels.pop(idx)

    def points_near_label(self, min_dist=None, max_dist=None, number=1):
        """
        Returns a random sample of coordinates which are in the vascinity of a
        label.  This operates on the current image and label.

        Argument min_dist is the minimum distance outwards from an edge of a 
                 label which will be sampled.  If not given or is None, all of 
                 labels interiors are sampled from.  If negative will sample
                 from inside of labels.  
        Argument max_dist is the maximum distance outwards from an edge of a 
                 label which will be sampled.  If not given or is None, 
                 unlabeled areas are sampled from.  If negative will discard
                 sample points near the edges of labels.  
        Argument number is the number of unique samples to take.

        Returns list of pairs of (x, y) coordinates
        """
        # Load the label data
        lbl = np.asarray(Image.open(self.current_label))
        # Select the labels we are looking for (currently any of them)
        lbl = np.sum(lbl, axis=2) != 0

        dilate = scipy.ndimage.binary_dilation
        erode  = scipy.ndimage.binary_erosion

        if max_dist is None:    # No maximum distance
            within_max = np.ones_like(lbl)
        elif max_dist > 0:      # Dilate the labels
            within_max = dilate(lbl, iterations=max_dist)
        elif max_dist < 0:      # Erode the labels
            within_max = erode(lbl, iterations=-max_dist)

        if min_dist is None:    # No minimum distance
            within_min = np.zeros_like(lbl)
        elif min_dist > 0:      # Dilate the labels
            within_min = dilate(lbl, iterations=min_dist)
        elif min_dist < 0:      # Erode the labels
            within_min = erode(lbl, iterations=-min_dist)
        within_min = np.logical_not(within_min)

        sample_space = np.logical_and(within_max, within_min)

        if False:
            import matplotlib.pyplot as plt
            plt.figure('DEBUG points_near_label')
            plt.subplot(1,2,1)
            plt.imshow(Image.open(self.current_image))
            plt.title("Image")
            plt.subplot(1,2,2)
            plt.imshow(sample_space, interpolation='nearest')
            plt.title("Sample Space (In Red)")
            plt.show()

        # Unique samples
        nonz = np.transpose(np.nonzero(sample_space))
        samples = nonz[random.sample(range(nonz.shape[0]), number)]
        return [tuple(p) for p in samples]

    def label_at(self, coords):
        """
        This returns the name of the label at the given (X, Y) coordinates.

        Argument coords is either a pair of (X, Y) coordinates or an iterable
                 of pairs of (X, Y) coordinates.  Areas outside of the image are
                 considered unlabeled.

        Returns list of label names
        """
        if len(coords) == 2 and all(isinstance(x, int) for x in coords):
            # Argument coords is a single coordinate.  Wrap it in a list...
            coords = [coords]
        # Save the label image because this function gets called a lot of times.
        # I really should have made this a property instead of opening the label
        # image in every method.  
        label_path, label_image = getattr(self, '_label_at_cache', (None, None))
        if label_path is None or label_path != self.current_label:
            label_path = self.current_label
            label_image = np.asarray(Image.open(label_path))
            self._label_at_cache = (label_path, label_image)
        x_bounds = range(label_image.shape[0])
        y_bounds = range(label_image.shape[1])
        label_names = []
        for x, y in coords:
            if x not in x_bounds or y not in y_bounds:
                label_names.append('unlabeled')
            else:
                color = label_image[x, y, :]
                color = tuple(np.squeeze(color))    # Colors are 4-tuples of ints
                label_names.append( self.names[color] )
        return label_names

    def label_id(self, label):
        """
        Converts a label (either a color or name) into a constant & unique 
        identifier.  Label identifiers are integers in range(len(self.names)).
        """
        if isinstance(label, tuple):
            label = self.names[label]
        return self.sorted_names.index(label)

    def sample_labels(self, sample_points):
        """
        Takes a sampling of the labels in the current image.

        Argument sample_points is a list of (X, Y) coordinates to sample at.  

        Returns vector of label occurances, with one entry per label type. 
        """
        sample = np.zeros((len(self.names),), dtype=np.float)
        labels = self.label_at(sample_points)
        for lbl in labels:
            idx = self.label_id(lbl)
            sample[idx] += 1
        return sample

    def compare_label_samples(self, s1, s2):
        """
        Compares two samples of labels and returns a score in the range [0, 1]
        with 0 as no agreement and 1 as perfect agreement.

        Experimental: This ignores unlabeled data.
                    Use the other metric to test labeled-vs-unlableled estimates.
        """
        assert(s1.shape == s2.shape)
        s1_sum = np.sum(s1)
        s2_sum = np.sum(s2)
        if s1_sum == 0 or s2_sum == 0:
            # This means one of the things didn't output anything, probably a bug...
            return 0
        s1 = np.array(s1) / s1_sum
        s2 = np.array(s2) / s2_sum
        if True:
            # Discard the estimates of unlabeled input.  
            unlbl = self.unlabeled_index
            s1[unlbl] = 0
            s2[unlbl] = 0
        # How to compare PDFs?
        return np.sum(np.minimum(s1, s2))

    def compare_label_samples_background(self, s1, s2):
        assert(s1.shape == s2.shape)
        s1_sum = np.sum(s1)
        s2_sum = np.sum(s2)
        if s1_sum == 0 or s2_sum == 0:
            return 0 # This means one of the things didn't output anything, probably a bug...
        s1 = np.array(s1) / s1_sum
        s2 = np.array(s2) / s2_sum
        idx = self.sorted_names.index('unlabeled')
        return min(s1[idx], s2[idx]) / max(s1[idx], s2[idx])

    def _fix_alpha_channel(self):
        # This is a fix for a bug where the Alpha channel was dropped.
        colors3to4 = [(c[:3], c[3]) for c in self.names.keys()]
        colors3to4 = dict(colors3to4)
        assert(len(colors3to4) == len(self.names)) # Dropped alpha channel causes colors to collide :(
        for lbl in self.labels:
            if lbl is None:
                continue    # No label file created yet.
            img  = Image.open(lbl)
            size = img.size
            img  = np.array(img)
            if img.shape[2] == 4:
                continue    # Image has alpha channel, good.
            elif img.shape[2] == 3:
                # Lookup each (partial) color and find what its alpha should be.
                alpha   = np.apply_along_axis(lambda c: colors3to4[tuple(c)], 2, img)
                data    = np.dstack([img, np.array(alpha, dtype=np.uint8)])
                new_img = Image.frombuffer("RGBA", size, data, "raw", "RGBA", 0, 1)
                new_img.save(lbl)
                print("FIXED", lbl)

    def statistics(self):
        if self.path is None:
            return "No dataset loaded."

        s = ''
        # Determine how many of the images have meaningful labels.
        only_labeled = deepcopy(self)
        only_labeled.discard_unlabeled_data()
        s += 'Fraction of images which have labels: %d / %d = %d%%\n'%(
                len(only_labeled),
                len(self), int(round(100 * len(only_labeled) / len(self))))

        # Determine how return how many images each label appears in.
        label_histogram = dict((nm, 0) for nm in self.names.values())
        for label_path in self.labels:
            if label_path is None:
                continue
            label_image = np.array(Image.open(label_path))
            # Flatten X & Y dimensions, keep the color channel intact.
            label_image = label_image.reshape(-1, 4)
            labels_used = np.unique(label_image, axis=0)
            for label_color in labels_used:
                label_name = self.names[tuple(label_color)]
                label_histogram[label_name] += 1
        # Now put together a table to present this histogram.
        s += "\n"
        s += "Histogram of label occurances,\n"
        s += "Each label counts once per image it occurs in.\n"
        label_histogram = list(label_histogram.items())
        label_histogram.sort(key=lambda name_occur: name_occur[0])  # Sort by name
        max_name_length  = str(max(len(str(name))  for name, occur in label_histogram))
        max_occur_length = str(max(len(str(occur)) for name, occur in label_histogram))
        table_format = "{:.<" + max_name_length + "}...{:.>" + max_occur_length + '}\n'
        for name, occurances in label_histogram:
            s += table_format.format(name, occurances)

        return s

    def split_dataset(self, train, test=None, verbosity=1):
        """
        Randomly divide the images into testing and training image sets.

        Argument train and test are the fractions of images to put in each 
                 data-subset.  Numerics, they do not need to add up to any 
                 particular denominator.  If the second argument is not given, 
                 then the first argument is used as the number of test images.

        Optional argument verbosity ...

        Returns (train, test)
            Where train and test are a pair of Datasets instance.
        """
        # Determine how many and which images will go in each data-subset.
        if test is None:
            # Only one argument given, arg is named train but is intended as test.
            test = train
            if test < 1:    # User gave fraction of images to use.
                test = int(round(len(self) * test))
            else:           # User gave litteral number oif images to use.
                test = int(round(test))
        else:
            denom   = train + test
            test    = int(round(len(self) * test  / denom))
        test_index  = random.sample(range(len(self)), test)
        test_index  = set(test_index)
        # Make empty subsets.
        train_data = deepcopy(self)
        test_data  = deepcopy(self)
        train_data.images.clear()
        train_data.labels.clear()
        test_data.images.clear()
        test_data.labels.clear()
        # Move the images into their subsets.
        for idx, datum in enumerate(zip(self.images, self.labels)):
            img, lbl = datum
            if idx in test_index:
                test_data.images.append(img)
                test_data.labels.append(lbl)
            else:
                train_data.images.append(img)
                train_data.labels.append(lbl)

        if verbosity:
            print("Split dataset into (%d) train and (%d) test images"%
                                            (len(train_data), len(test_data)))
        return train_data, test_data

