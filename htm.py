# Written by David McDougall, 2017

import numpy as np
import math
import scipy.ndimage
import scipy.spatial
import scipy.stats
import random
import cv2
import pickle
import sys
import os, os.path
import re
import cv2
import PIL, PIL.ImageDraw
from genetics import Parameters
import htm_cython

def random_tag(length=2):
    return ''.join(chr(ord('A') + random.randrange(26)) for _ in range(length))
instance_tag = random_tag()


class SaveLoad:
    """
    Saves/Loads pickled classes from file

    Default folder is name of program + "_data/".  
    Change this by setting SaveLoad.tag to the desired postfix.  
    This tag may contain slashes, which of course indicate more directories.  

    Subclasses must define the class attribute 'data_extension' to the 
    default file name extension for the save files.
    """
    # TODO: Consider naming save files by their instance-tag instead of their age.  
    tag = 'data'
    def save(self, filename=None, extension=None):
        if filename is None:
            if extension is None:
                extension = self.data_extension
            program, ext  = os.path.splitext(sys.argv[0])
            path, program = os.path.split(program)
            folder        = os.path.join(path, program + '_' + SaveLoad.tag)
            filename      = program + '.' + str(self.age) + '.' + extension.lstrip('.')
            filename      = os.path.join(folder, filename)
        print("Saving", filename)
        try:
            os.makedirs(folder)
        except OSError:
            pass
        pickle.dump(self, open(filename, 'wb'))

    @classmethod
    def load(cls, filename=None, extension=None):
        if filename is None:
            if extension is None:
                extension = cls.data_extension
            program, ext  = os.path.splitext(sys.argv[0])
            path, program = os.path.split(program)
            folder        = os.path.join(path, program + '_' + SaveLoad.tag)
            matches       = []
            for fn in os.listdir(folder):
                fn_format = program + r'\.(\d+)\.' + extension.lstrip('.') + '$'
                m = re.match(fn_format, fn)
                if m:
                    age = int(m.groups()[0])
                    matches.append((fn, age))
            matches.sort(key=lambda p: p[1])    # Sort by age
            if not matches:
                raise FileNotFoundError("No file in directory %s/ matching r'%s'"%(folder, fn_format))
            filename = matches[-1][0]
            filename = os.path.join(folder, filename)
        self = pickle.load(open(filename, 'rb'))
        print("Loaded %s   age %d"%(filename, self.age))
        return self


# TODO: This should use or at least print the radius, ie the distance at which
# two numbers will have 50% overlap.  Radius is a replacement for resolution.
class RandomDistributedScalarEncoderParameters(Parameters):
    parameters = [
        "resolution",
        "size",
        "sparsity",     # TODO: Replace this with 'on_bits'
    ]
    def __init__(self, resolution = 1, size = 128, sparsity = .15):
        self.resolution   = resolution
        self.size         = size
        self.sparsity     = sparsity

class RandomDistributedScalarEncoder:
    """https://arxiv.org/pdf/1602.05925.pdf"""
    def __init__(self, parameters):
        self.args = args    = parameters
        self.size           = int(round(args.size))
        self.on_bits        = int(round(args.size * args.sparsity))
        self.output_shape   = (int(round(self.size)),)

    def encode(self, value):
        # This must be integer division! Everything under the resolution must be removed.
        index = value // self.args.resolution
        code = np.zeros(self.output_shape, dtype=np.bool)
        for offset in range(self.on_bits):
            # Cast to string before hash, python3 will not hash an integer, uses
            # value instead.
            h = hash(str(index + offset))
            bucket = h % self.size
            # If this bucket is already full, walk around until it finds one
            # that isn't taken.
            while code[bucket]:
                bucket = (bucket + 1) % self.size
            code[bucket] = True
        return code


class EnumEncoder:
    # TEST ME AND USE ME OR REMOVE ME
    """
    Encodes arbirary enumerated values.
    There is no semantic similarity between encoded values.
    """
    def __init__(self, bits, sparsity, diag=True):
        self.bits         = bits
        self.sparsity     = sparsity
        self.on_bits      = int(round(self.bits * self.sparsity))
        self.enums        = set()
        self.output_shape = (self.bits,)
        if diag:
            print("Enum Encoder: %d bits %.2g%% sparsity"%(bits, 100*sparsity))

    def add_enum(self, names):
        """Accepts either a string or a list of strings."""
        if isinstance(names, str):
            self.enums.add(names)
        else:
            self.enums.update(names)

    def encode(self, names):
        """
        Accepts either a string of a list of strings.
        All enum names must have been added via ee.add_enum(...)

        Returns dense boolean array, union of the given names
        """
        if isinstance(names, str):
            names = [names]
        bits = np.zeros((self.bits,), dtype=np.bool)
        bits_per_enum = int(round(self.bits * self.sparsity / len(names)))
        total_bits = len(names) * bits_per_enum
        for nm in names:
            assert(nm in self.enums)
            r = random.Random(hash(nm))
            b = r.sample(range(self.bits), total_bits)
            b = random.sample(b, bits_per_enum)
            bits[b] = True
        return bits


class BWImageEncoder:
    def __init__(self, input_space, diag=True):
        input_space = tuple(input_space)
        self.output_shape = input_space + (2,)
        if diag:
            print("Image Encoder")
            print("\tInput -> Output shapes are", input_space, '->', self.output_shape)

    def encode(self, image):
        mean = np.mean(image)
        on_bits  = image >= mean
        off_bits = np.logical_not(on_bits)
        return np.dstack([on_bits, off_bits])


class ChannelEncoderParameters(Parameters):
    parameters = [
        'num_samples',
        'sparsity',
    ]
    def __init__(self, num_samples = 5, sparsity = .20,):
        """
        Argument num_samples is number of bits in the output SDR which will
                 represent each input number, this is the added data depth.
        Argument sparsity is fraction of output which on average will be active.
                 This is also the fraction of the input spaces which (on 
                 average) each range covers.
        """
        self.num_samples  = num_samples
        self.sparsity     = sparsity

# TODO: Measure the output sparsity
class ChannelEncoder:
    """
    This assigns a random range to each bit of the output SDR.  Each bit becomes
    active if its corresponding input falls in its range.  By using random
    ranges, each bit represents a different thing even if it mostly overlaps
    with other comparable bits.  This way redundant bits add meaning.
    """
    def __init__(self, parameters, input_shape,
        dtype       = np.float64,
        drange      = range(0,1),
        wrap        = False):
        """
        Argument parameters is an instance of ChannelEncoderParameters.
        Argument input_shape is tuple of dimensions of each input frame.
        Argument dtype is numpy data type of channel.
        Argument drange is a range object or a pair of values representing the 
                 range of possible channel values.
        Argument wrap ... default is False.
                 This supports modular input spaces and ranges which wrap
                 around. It does this by rotating the inputs by a constant
                 random amount which hides where the discontinuity in ranges is.
                 No ranges actually wrap around the input space.
        """
        assert(isinstance(parameters, ChannelEncoderParameters))
        self.args = args  = parameters
        self.input_shape  = tuple(input_shape)
        self.output_shape = self.input_shape + (int(round(args.num_samples)),)
        self.dtype        = dtype
        self.drange       = drange
        self.len_drange   = max(drange) - min(drange)
        self.wrap         = bool(wrap)
        if self.wrap:
            self.offsets  = np.random.uniform(0, self.len_drange, self.input_shape)
            self.offsets  = np.array(self.offsets, dtype=self.dtype)
        # Mean and std-dev of sizes of ranges which bits respond to.
        self.mean_size    = self.len_drange * args.sparsity
        self.std_size     = self.mean_size / 8
        # Make the centers and sizes of each range.
        if self.wrap:
            # If wrapping is enabled then don't generate ranges which are very
            # likely to get truncated near the edges.
            centers = np.random.uniform(min(self.drange) + self.std_size,
                                        max(self.drange) - self.std_size,
                                        size=self.output_shape)
        else:
            centers = np.random.uniform(min(self.drange),
                                        max(self.drange),
                                        size=self.output_shape)
        sizes = np.random.normal(self.mean_size, self.std_size, self.output_shape) / 2
        # Make the lower and upper bounds of the ranges.
        self.low  = np.array(centers - sizes, dtype=self.dtype)
        self.high = np.array(centers + sizes, dtype=self.dtype)

    def encode(self, img):
        assert(img.shape == self.input_shape)
        assert(img.dtype == self.dtype)
        if self.wrap:
            img += self.offsets
            # Technically this should subtract min(drange) before doing modulus
            # but the results should also be indistinguishable B/C of the random
            # offsets.  Min(drange) effectively becomes part of the offset.
            img %= self.len_drange
            img += min(self.drange)
        img = img.reshape(img.shape + (1,))
        return np.logical_and(self.low <= img, img <= self.high)

    def __str__(self):
        lines = ["Channel Encoder,  num-samples %d"%int(round(self.args.num_samples))]
        lines.append("\tSparsity %.03g, Deviation %.03g, %s %s %s"%(
                self.args.sparsity,
                self.std_size,
                self.dtype.__name__,
                self.drange,
                'Wrapped' if self.wrap else ''))
        return '\n'.join(lines)


class ChannelThresholderParameters(Parameters):
    parameters = [
        'channel',
        'mean',
        'stddev',
    ]
    def __init__(self,
        channel = None,
        mean    = .5,
        stddev  = .2,):
        """
        Argument channel is an instance of ChannelEncoderParameters
        Argument mean is the average activation threshold.
        Argument stddev is the standard deviation of activation thresholds.
        """
        if channel is not None:
            self.channel = channel
        else:
            self.channel = ChannelEncoderParameters()
        self.mean    = mean
        self.stddev  = stddev

# TODO: Measure the output sparsity
class ChannelThresholder:
    """
    Assigns each bit of the given channel encoder with an additional activation
    threshold.  A bit becomes active iff the underlying channel encoder
    activates it and its magnitude is not less than its threshold.  Activation
    thresholds are normally distributed.

    This class wraps a channel encoder.
    """
    def __init__(self, parameters, input_shape,  dtype, drange, wrap):
        """
        Argument parameters is an instance of ChannelThresholderParameters.
        Argument input_shape is tuple of dimensions of each input frame.
        Arguments dtype, drange, and wrap are passed through to the underlying
                  channel encoder.
        """
        assert(isinstance(parameters, ChannelThresholderParameters))
        self.args = args = parameters
        self.channel     = ChannelEncoder(args.channel, input_shape, 
                            dtype=dtype, drange=drange, wrap=wrap)
        self.thresholds  = np.random.normal(args.mean, args.stddev, self.channel.input_shape)
        self.thresholds  = np.array(self.thresholds, dtype)
        self.output_shape = self.channel.output_shape

    def encode(self, img_data, magnitude):
        """
        Send raw data and magnitudes, this runs the channel encoder as well as
        the thresholder.
        """
        sdr = self.channel.encode(img_data)
        assert(magnitude.shape == self.channel.input_shape)
        sdr[magnitude < self.thresholds] = False
        return sdr


class EyeSensorParameters(Parameters):
    parameters = [
        # Retina Parameters
        'eye_dimensions',
        'fovea_param_1',
        'fovea_param_2',
        'min_scale',
        'max_scale',
        'hue_encoder',
        'sat_encoder',
        'val_encoder',
        'edge_encoder',
        # Control Vector Parameters
        # 'num_cv',
        # 'pos_stddev',
        # 'angle_stddev',
        # 'scale_stddev',
        # Motor Sensor Parameters
        # 'position_encoder',
        # 'velocity_encoder',
        # 'angle_encoder',
        # 'angular_velocity_encoder',
        # 'scale_encoder',
        # 'scale_velocity_encoder',
    ]
    def __init__(self,
        # Retina Defaults
        eye_dimensions  = (512, 512),
        fovea_param_1   = .05,
        fovea_param_2   = 20,
        min_scale       =  1,
        max_scale       = 10,
        hue_encoder     = None,
        sat_encoder     = None,
        val_encoder     = None,
        edge_encoder    = None,
        # Control Vector Defaults
        # num_cv       = 600,
        # pos_stddev   = 10,
        # angle_stddev = math.pi / 8,
        # scale_stddev = 2,
        # Motor Sensor Defaults
        # position_encoder = RandomDistributedScalarEncoderParameters(
        #                         size        = 100,
        #                         sparsity    = 0.20,
        #                         resolution  = 1,),
        # velocity_encoder = RandomDistributedScalarEncoderParameters(
        #                         size        = 100,
        #                         sparsity    = 0.20,
        #                         resolution  = 1,),
        # angle_encoder = RandomDistributedScalarEncoderParameters(
        #                         size        = 100,
        #                         sparsity    = 0.20,
        #                         resolution  = math.pi / 80,),
        # angular_velocity_encoder = RandomDistributedScalarEncoderParameters(
        #                         size        = 100,
        #                         sparsity    = 0.20,
        #                         resolution  = math.pi / 80,),
        # scale_encoder = RandomDistributedScalarEncoderParameters(
        #                         size        = 100,
        #                         sparsity    = 0.20,
        #                         resolution  = .2,),
        # scale_velocity_encoder = RandomDistributedScalarEncoderParameters(
        #                         size        = 100,
        #                         sparsity    = 0.20,
        #                         resolution  = .2, ),
        ):
        """
        Argument eye_dimensions ...
        Arguments fovea_param_1 and fovea_param_2 ...
        Arguments min_scale and max_scale ...
        Arguments hue_encoder, sat_encoder and val_encoder are instances of
                  ChannelEncoderParameters.
        Argument edge_encoder is an instance of ChannelThresholderParameters.

        Argument num_cv is the approximate number of control vectors to use.
        Arguments pos_stddev, angle_stddev, and scale_stddev are the standard
                  deviations of the control vector movements, they are normally
                  distributed about a mean of 0.

        Arguments position_encoder, velocity_encoder, angle_encoder, angular_velocity_encoder,
                  scale_encoder, and scale_velocity_encoder are instances of 
                  RandomDistributedScalarEncoderParameters.
        """
        # Get the parent class to save all these parameters.
        kw_args = locals().copy()
        del kw_args['self']
        del kw_args['__class__']
        super().__init__(**kw_args)
        if hue_encoder is None:
            self.hue_encoder     = ChannelEncoderParameters()
        if sat_encoder is None:
            self.sat_encoder     = ChannelEncoderParameters()
        if val_encoder is None:
            self.val_encoder     = ChannelEncoderParameters()
        if edge_encoder is None:
            self.edge_encoder    = ChannelThresholderParameters()

class EyeSensor:
    """
    Eye sensor with controllable movement and central fovae.

    This eye sensor was designed with the following criteria:
    1) The central fovae should be capable of identifying objects in 1 or 2 saccades.
    2) The periferal vision should be capable of locating and tracking objects.

    This eye has 4 degrees of freedom: X and Y location, scale, and orientation.
    These values can be controlled by activating control vectors, each of which 
    has a small but cumulative effect.  CV's are normally distributed with a
    mean of zero.  

    The eye outputs the its current location, scale and orientation as well as 
    their first derivatives w/r/t time as a dense SDR.

    Fun Fact: The human optic nerve has 800,000 ~ 1,700,000 nerve fibers.
    """
    def __init__(self, parameters):
        """
        Attributes:
            The following are the shapes of SDR's used by this class
                self.view_shape         retina's output
                self.control_shape      eye movement input controls
                self.motor_shape        internal motor sensor output

            self.rgb            The most recent view, kept as a attribute for
                                making diagnostics.

            self.motor_sdr      The SDR encoded output of the internal motor 
                                sensors, kept as a attribute for convenience.

            self.position       (X, Y) coords of eye within image, Read/Writable
            self.orientation    ... Read/Writable
            self.scale          ... Read/Writable

            self.gaze   List of tuples of (X, Y, Orientation, Scale)
                        History of recent movements, self.move() updates this.
                        This is cleared by the following methods:
                            self.reset()
                            self.new_image()
                            self.center_view()
                            self.randomize_view()

        Private Attributes:
            self.eye_coords.shape = (2, view-x, view-y)
            self.eye_coords[input-dim, output-coordinate] = input-coordinate
        """
        self.args = args = parameters
        self.eye_dimensions = tuple(int(round(ed)) for ed in args.eye_dimensions)
        self.eye_coords = EyeSensor.complex_eye_coords(self.eye_dimensions,
                                        args.fovea_param_1, args.fovea_param_2)
        self.hue_encoder = ChannelEncoder(  args.hue_encoder,
                                            input_shape = self.eye_dimensions,
                                            dtype       = np.float32,
                                            drange      = range(0,360),
                                            wrap        = True,)
        self.sat_encoder = ChannelEncoder(  args.sat_encoder,
                                            input_shape = self.eye_dimensions,
                                            dtype  = np.float32,
                                            drange = (0, 1),
                                            wrap   = False,)
        self.val_encoder = ChannelEncoder(  args.val_encoder,
                                            input_shape = self.eye_dimensions,
                                            dtype  = np.float32,
                                            drange = (0, 1),
                                            wrap   = False,)
        self.edge_encoder = ChannelThresholder(args.edge_encoder,
                                            input_shape = self.eye_dimensions,
                                            dtype  = np.float32,
                                            drange = (-math.pi, math.pi),
                                            wrap   = True)

        depth = sum((self.hue_encoder.output_shape[2],
                     self.sat_encoder.output_shape[2],
                     self.val_encoder.output_shape[2],
                     self.edge_encoder.output_shape[2],))
        self.view_shape = self.eye_dimensions + (depth,)

        # self.control_vectors, self.control_shape = self.make_control_vectors(
        #         num_cv       = args.num_cv,
        #         pos_stddev   = args.pos_stddev,
        #         angle_stddev = args.angle_stddev,
        #         scale_stddev = args.scale_stddev,
        #         verbosity    = verbosity,)

        # size    = 128
        # on_bits = .10 * size
        # self.motor_encoders, self.motor_shape = self.make_motor_encoders(size, on_bits)

        self.reset()
        # if verbosity:
            # self.print_parameters()

    @staticmethod
    def simple_eye_coords(eye_dims):
        """
        Returns sampling coordinates for a uniform density eye.
        Argument eye_dims is shape of eye receptors, output shape
        """
        return np.mgrid[[slice(-d//2, d//2) for d in eye_dims]]

    @staticmethod
    def complex_eye_coords(eye_dims, fovea_param_1, fovea_param_2, verbosity=0):
        """
        Returns sampling coordinates for a non-uniform density eye.
            retval[output-coord] = sample-offset

        Argument eye_dims is shape of eye receptors, output shape
        Arguments fovea_param_1 and fovea_param_2 are magic constants, try 0.05
                  and 20, respctively.
        """
        def gauss(x, mean, stddev):
            return np.exp(-(x - mean) ** 2 / (2 * stddev ** 2))

        # Flat eye is index array of the output locations
        flat_eye    = EyeSensor.simple_eye_coords(eye_dims)
        flat_eye    = np.array(flat_eye, dtype=np.float64)    # Cast to float
        # Jitter each coordinate, but not enough to make them cross.
        # The purpose of this jitter is to break up any aliasing patterns.
        flat_eye    += np.random.normal(0, .33, flat_eye.shape)
        # Radial distances from center to output locations.
        radius      = np.hypot(flat_eye[0], flat_eye[1])
        max_radius  = int(np.ceil(np.max(radius))) + 1

        #
        # Density function
        # This controls the shape of the eye.
        #
        density = [fovea_param_1 + gauss(x, 0, max_radius/fovea_param_2) for x in range(max_radius)]

        # Force Density[radius == 0] == 0.
        # This is needed for interpolation to work.
        density = [0] + density

        # Integrate density over radius and as a function of radius.
        # Units are receptors per unit radial distance
        retina_area = np.cumsum(density)
        # Normalize density's integral to 1, this is needed for jitter.
        density = np.divide(density, retina_area[-1])
        # Normalize number of receptors to range [0, max-radius]
        retina_area *= max_radius / retina_area[-1]
        # Invert, units are now units radial distance per receptor.
        inverse = scipy.interpolate.interp1d(retina_area, np.arange(max_radius + 1))
        receptor_radius = inverse(np.arange(max_radius))

        # receptor_radius is mapping from output-space radius to input-space
        # radius.  Apply it to the flat coordinates to find eye coordinates.
        radius_idx = np.array(np.rint(radius), dtype=np.int) # Integer cast radius for use as index.
        flat_eye[:, ...] *= np.nan_to_num(receptor_radius[radius_idx] / radius)

        if verbosity >= 2:
            plt.figure("Complex Eye Diagnostics")
            plt.subplot(1, 2, 1)
            plt.plot(density)
            plt.title("Density")
            plt.ylabel("Fraction of receptors")
            plt.xlabel("Distance from center")
            plt.subplot(1, 2, 2)
            plt.plot(receptor_radius)
            plt.title("Receptor Mapping")
            plt.ylabel("Input Radius")
            plt.xlabel("Output radius")
            plt.show()
        return flat_eye

    @staticmethod
    def make_control_vectors(num_cv, pos_stddev, angle_stddev, scale_stddev, verbosity=0):
        """
        Argument num_cv is the approximate number of control vectors to create
        Arguments pos_stddev, angle_stddev, and scale_stddev are the standard
                  deviations of the controls effects of position, angle, and 
                  scale.

        Returns pair of control_vectors, control_shape

        The control_vectors determines what happens for each output. Each
        control is a 4-tuple of (X, Y, Angle, Scale) movements. To move,
        active controls are summed and applied to the current location.
        """
        cv_sz = int(round(num_cv // 6))
        control_shape = (6*cv_sz,)

        pos_controls = [
            (random.gauss(0, pos_stddev), random.gauss(0, pos_stddev), 0, 0)
                for i in range(4*cv_sz)]

        angle_controls = [
            (0, 0, random.gauss(0, angle_stddev), 0)
                for angle_control in range(cv_sz)]

        scale_controls = [
            (0, 0, 0, random.gauss(0, scale_stddev))
                for scale_control in range(cv_sz)]

        control_vectors = pos_controls + angle_controls + scale_controls
        random.shuffle(control_vectors)
        control_vectors = np.array(control_vectors)

        # Add a little noise to all control vectors
        control_vectors[:, 0] += np.random.normal(0, pos_stddev/10,    control_shape)
        control_vectors[:, 1] += np.random.normal(0, pos_stddev/10,    control_shape)
        control_vectors[:, 2] += np.random.normal(0, angle_stddev/10,  control_shape)
        control_vectors[:, 3] += np.random.normal(0, scale_stddev/10,  control_shape)
        if verbosity >= 2:
            print("Control Vectors")
            print(control_vectors)
        return control_vectors, control_shape

    def make_motor_encoders(self, size, on_bits):
        """
        Create the Motor Sensor Encoders
        This creates eight (8) encoders, all with the same size and on-bits.

        Argument size
        Argument on_bits

        Returns encoders, shape
            Where encoders is a list of RDSE's
              and shape is the resulting motor sensor SDR shape
        """
        dp      = self.pos_stddev / 10
        ds      = self.scale_stddev / 10
        da      = self.angle_stddev / 10
        motor_encoders = [                  # (Res, Size, OnBits)
            htm.RandomDistributedScalarEncoder(dp, size, on_bits),    # x position
            htm.RandomDistributedScalarEncoder(dp, size, on_bits),    # y position
            htm.RandomDistributedScalarEncoder(da, size, on_bits),    # orientation
            htm.RandomDistributedScalarEncoder(ds, size, on_bits),    # scale

            htm.RandomDistributedScalarEncoder(dp, size, on_bits),    # x velocity
            htm.RandomDistributedScalarEncoder(dp, size, on_bits),    # y velocity
            htm.RandomDistributedScalarEncoder(da, size, on_bits),    # angular velocity
            htm.RandomDistributedScalarEncoder(ds, size, on_bits),    # scale velocity
        ]
        motor_shape = (sum(enc.output_shape[0] for enc in motor_encoders),)
        return motor_encoders, motor_shape

    def reset(self):
        self.position    = (0,0)
        self.orientation = 0
        self.scale       = 0
        # self.motor_sdr   = np.zeros(self.motor_shape, dtype=np.bool)
        self.image       = None
        self.gaze        = []
        self.edges       = None

    def randomize_view(self):
        """Set the eye's view point to a random location"""
        self.orientation = random.random() * 2 * math.pi
        self.scale       = random.uniform(self.args.min_scale, self.args.max_scale)
        eye_radius       = np.multiply(self.scale / 2, self.eye_dimensions)
        self.position    = [np.random.uniform(0, dim) for dim in self.image.shape[:2]]
        # Discard any prior gaze tracking after forcibly moving eye position to new starting position.
        self.gaze = [tuple(self.position) + (self.orientation, self.scale)]

    def center_view(self):
        """Center the view over the image"""
        self.orientation = 0
        self.position = np.divide(self.image.shape[:2], 2)
        self.scale = np.max(np.divide(self.image.shape[:2], self.eye_dimensions))
        # Discard any prior gaze tracking after forcibly moving eye position to new starting position.
        self.gaze = [tuple(self.position) + (self.orientation, self.scale)]

    def new_image(self, image, diag=False):
        if isinstance(image, str):
            self.image_file = image
            self.image = np.array(PIL.Image.open(image))
        else:
            self.image_file = None
            self.image = image
        # Get the image into the right format.
        if self.image.dtype != np.uint8:
            raise TypeError('Image %s dtype is not unsigned 8 bit integer, image.dtype is %s.'%(
                    '"%s"'%self.image_file if self.image_file is not None else 'argument',
                    self.image.dtype))
        self.image = np.squeeze(self.image)
        if len(self.image.shape) == 2:
            self.image = np.dstack([self.image] * 3)

        self.preprocess_edges()
        self.randomize_view()

        if diag:
            plt.figure('Image')
            plt.title('Image')
            plt.imshow(self.image, interpolation='nearest')
            plt.show()

    def preprocess_edges(self):
        # Calculate the sobel edge features
        grey    = np.sum(self.image/255., axis=2, keepdims=False, dtype=np.float64)/3.
        sobel_x = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=7)
        sobel_y = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=7)
        self.edge_angles    = np.arctan2(sobel_y, sobel_x)
        self.edge_angles    = np.array(self.edge_angles, dtype=np.float32)
        self.edge_magnitues = (sobel_x ** 2 + sobel_y ** 2) ** .5
        self.edge_magnitues = np.array(self.edge_magnitues, dtype=np.float32)

    def move(self, control_index):
        """
        Apply the given control vector to the current gaze location
        Also calculates the real velocity in each direction

        Argument control_index is an index array of ON-bits in the control space.

        Returns and SDR encoded representation of the eyes velocity.
        """
        # Calculate the forces on the motor
        controls = self.control_vectors[control_index]
        controls = np.sum(controls, axis=0)
        dx, dy, dangle, dscale = controls
        # Calculate rotation effects
        self.orientation = (self.orientation + dangle) % (2*math.pi)
        # Calculate scale effects
        new_scale  = np.clip(self.scale + dscale, self.min_scale, self.max_scale)
        real_ds    = new_scale - self.scale
        avg_scale  = (new_scale + self.scale) / 2
        self.scale = new_scale
        # Calculate position effectsdy
        # EXPERMENTIAL: Scale the movement such that the same CV yields the same
        # visual displacement, regardless of scale.
        x, y     = self.position
        dx       *= avg_scale   
        dy       *= avg_scale   
        p        = [x + dx, y + dy]
        p        = np.clip(p, [0,0], self.image.shape[:2])
        real_dp  = np.subtract(p, self.position)
        self.position = p
        # Put together information about the motor
        velocity = (
            self.position[0],
            self.position[1],
            self.orientation,
            self.scale,
            real_dp[0],
            real_dp[1],
            dangle,
            real_ds,
        )
        self.gaze.append(tuple(self.position) + (self.orientation, self.scale))
        # Encode the motors sensory states and concatenate them into one big SDR.
        v_enc = [enc.encode(v) for v, enc in zip(velocity, self.motor_encoders)]
        self.motor_sdr = np.concatenate(v_enc)
        # Shuffle the SDR with a psuedo random permutation seeded with the number 42
        pass
        return self.motor_sdr

    def view(self):
        """
        Returns the image which the eye is currently seeing.

        Attribute self.rgb is set to the current image which the eye is seeing.
        """
        # Rotate the samples points
        c   = math.cos(self.orientation)
        s   = math.sin(self.orientation)
        rot = np.array([[c, -s], [s, c]])
        global_coords = self.eye_coords.reshape(self.eye_coords.shape[0], -1)
        global_coords = np.matmul(rot, global_coords)
        # Scale/zoom the sample points
        global_coords *= self.scale
        # Position the sample points
        global_coords += np.array(self.position).reshape(2, 1)
        global_coords = tuple(global_coords)

        # Extract the view from the larger image
        channels = []
        for c_idx in range(3):
            ch = scipy.ndimage.map_coordinates(self.image[:,:,c_idx], global_coords,
                                            mode='constant',    # No-wrap, fill
                                            cval=255,           # Fill value
                                            order=3)
            channels.append(ch.reshape(self.eye_dimensions))
        self.rgb = rgb = np.dstack(channels)

        # Convert view to HSV and encode HSV to SDR.
        hsv         = np.array(rgb, dtype=np.float32)
        hsv         /= 255.
        hsv         = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)
        hue_sdr     = self.hue_encoder.encode(hsv[..., 0])
        sat_sdr     = self.sat_encoder.encode(hsv[..., 1])
        val_sdr     = self.val_encoder.encode(hsv[..., 2])

        # Extract edge samples
        angles = scipy.ndimage.map_coordinates(self.edge_angles, global_coords,
                                        mode='constant',    # No-wrap, fill
                                        cval=0,             # Fill value
                                        order=0)            # Take nearest value, no interp.
        mags = scipy.ndimage.map_coordinates(self.edge_magnitues, global_coords,
                                        mode='constant',    # No-wrap, fill
                                        cval=0,             # Fill value
                                        order=3)
        angles   = angles.reshape(self.eye_dimensions)
        mags     = mags.reshape(self.eye_dimensions)
        edge_sdr = self.edge_encoder.encode(angles, mags)

        return np.dstack([hue_sdr, sat_sdr, val_sdr, edge_sdr])

    def input_space_sample_points(self, samples_size=100):
        """
        Returns a list of pairs of (x,y) coordinates into the input image which
        correspond to a uniform sampling of the receptors given the eyes current
        location.

        The goal with this method is to determine what the eye is currently
        looking at, given labeled training data.  There can be multiple labels
        in the eyes receptive field (read: input space) at once.  Also the
        labels are not perfectly accurate as they were hand drawn.  Taking a
        large sample of points is a good approximation for sampling all points
        in the input space, which would be the perfectly accurate solution (but
        would be slow and perfection is not useful).
        """
        # Rotate the samples points
        c   = math.cos(self.orientation)
        s   = math.sin(self.orientation)
        rot = np.array([[c, -s], [s, c]])
        global_coords = self.eye_coords.reshape(self.eye_coords.shape[0], -1)
        global_coords = np.matmul(rot, global_coords)

        # Scale/zoom the sample points
        global_coords *= self.scale

        # Position the sample points
        global_coords += np.array(self.position).reshape(2, 1)
        sample_index  = random.sample(range(global_coords.shape[1]), samples_size)
        samples       = global_coords[:, sample_index]
        return np.array(np.rint(np.transpose(samples)), dtype=np.int32)

    def gaze_tracking(self, diag=True):
        """
        Returns vector of tuples of (position-x, position-y, orientation, scale)
        """
        if diag:
            im   = PIL.Image.fromarray(self.image)
            draw = PIL.ImageDraw.Draw(im)
            width, height = im.size
            # Draw a red line through the centers of each gaze point
            for p1, p2 in zip(self.gaze, self.gaze[1:]):
                x1, y1, a1, s1 = p1
                x2, y2, a2, s2 = p2
                assert(x1 in range(0, width))
                assert(x2 in range(0, width))
                assert(y1 in range(0, height))
                assert(y2 in range(0, height))
                draw.line((x1, y1, x2, y2), fill='black', width=7)
                draw.line((x1, y1, x2, y2), fill='red', width=2)
            # Draw the bounding box of the eye sensor around each gaze point
            for x, y, orientation, scale in self.gaze:
                # Find the four corners of the eye's window
                corners = []
                for ec_x, ec_y in [(0,0), (0,-1), (-1,-1), (-1,0)]:
                    corners.append(self.eye_coords[:, ec_x, ec_y])
                # Rotate the corners
                c = math.cos(orientation)
                s = math.sin(orientation)
                rot = np.array([[c, -s], [s, c]])
                corners = np.transpose(corners)     # Convert from list of pairs to index array.
                corners = np.matmul(rot, corners)
                # Scale/zoom the sample points
                corners *= scale
                # Position the sample points
                corners += np.array([x, y]).reshape(2, 1)
                # Convert from index array to list of coordinates pairs
                corners = list(tuple(coord) for coord in np.transpose(corners))
                # Draw the points
                for start, end in zip(corners, corners[1:] + [corners[0]]):
                    assert(start[0] in range(0, width))
                    assert(start[1] in range(0, height))
                    assert(end[0]   in range(0, width))
                    assert(end[1]   in range(0, height))
                    draw.line(start+end, fill='green', width=2)
            del draw
            plt.figure("Gaze Tracking")
            im = np.array(im)
            plt.imshow(im, interpolation='nearest')
            plt.show()
        return self.gaze[:]

    def print_parameters(self):
        print("Eye Parameters")
        print("\tRetina Input -> Output Shapes %s -> %s"%(str(self.eye_dimensions), str(self.view_shape)))
        print("\tMotor Sensors %d, Motor Controls %d"%(self.motor_shape[0], self.control_shape[0]))

class EyeSensorSampler:
    """
    Samples eyesensor.rgb, the eye's view.

    Attribute samples is list of RGB numpy arrays.
    """
    def __init__(self, eyesensor, sample_period, number_of_samples=30):
        """
        This draws its samples directly from the output of eyesensor.view() by
        wrapping the method.
        """
        self.sensor      = sensor = eyesensor
        self.sensor_view = sensor.view
        self.sensor.view = self.view
        self.age         = 0
        self.samples     = []
        self.schedule    = random.sample(range(sample_period), number_of_samples)
        self.schedule.sort(reverse=True)
    def view(self, *args, **kw_args):
        """Wrapper around eyesensor.view which takes samples"""
        retval = self.sensor_view(*args, **kw_args)
        if self.schedule and self.age == self.schedule[-1]:
            self.schedule.pop()
            self.samples.append(np.array(self.sensor.rgb))
        self.age += 1
        return retval
    def view_samples(self):
        if not self.samples:
            return  # Nothing to show...
        plt.figure("Sample views")
        num = len(self.samples)
        rows = math.floor(num ** .5)
        cols = math.ceil(num / rows)
        for idx, img in enumerate(self.samples):
            plt.subplot(rows, cols, idx)
            plt.imshow(img, interpolation='nearest')
        plt.show()


class SynapseManager:
    """For pyramidal neuron synapses."""
    def __init__(self, input_dimensions, output_dimensions,
        radii=None,
        potential_pool=.95,
        coincidence_inc = 0.1,          # Do not touch.
        coincidence_dec = 0.02,         # Do not touch.
        permanence_threshold = 0.5,     # Do not touch.
        diag=True):
        """
        Argument input_dimensions is tuple of input space dimensions
        Argument output_dimensions is tuple of output space dimensions
        Argument radii is tuple of convolutional radii, must be same length as output_dimensions
                 radii units are the input space units
                 radii is optional, if not given assumes no topology
        Argument potential_pool is the fraction of possible inputs to include in 
                 each columns potential pool of input sources.  Default is .95

        Arguments coincidence_inc, coincidence_dec, permanence_threshold
                In theory, permanence updates are the amount of time it takes to
                learn  a thing, with units of number of gazes.

        If output_dimensions is shorter than input_dimensions then the trailing
        input_dimensions are not convolved over, are instead broadcast to all
        outputs which are connected via the convolution in the other dimensions.
        """
        self.input_dimensions     = tuple(input_dimensions)
        self.output_dimensions    = tuple(output_dimensions)
        self.num_outputs          = np.product(self.output_dimensions)
        self.coincidence_inc      = coincidence_inc
        self.coincidence_dec      = coincidence_dec
        self.permanence_threshold = permanence_threshold

        if diag:
            if isinstance(diag, str):
                print(diag, "Synapse Parameters")
            else:
                print("Synapse Parameters")
            print("\tInput -> Output shapes are",
                            self.input_dimensions, '->', self.output_dimensions)

            # TODO: Multiply both sides of the coincidence ratio by their
            # greatest common demonimator and then have both sides of the ratio
            # labeled as "Presynaptic coincidence threshold:  X Active : Y
            # Silent" And then view the scale differently. What I should really
            # do is use the ratio and scale as parameters and calculate the inc
            # and dec.
            coincidence_ratio = self.coincidence_inc / self.coincidence_dec
            print('\tCoincidence Ratio', self.coincidence_inc, '/',
                        self.coincidence_dec, '=', coincidence_ratio)

        # Both inputs and outputs are identified by their flat-index, which is
        # their index into their space after it's been flattened.  All outputs
        # have the same number of inputs in their potential pool.

        # self.sources[output-index][input-number] = input-index
        # self.sources.shape = (num_outputs, num_inputs)
        if radii is not None:
            self.radii = radii
            self.normally_distributed_connections(input_dimensions, output_dimensions, potential_pool, radii)
            if diag:
                print("\tDensity within 1/2/3 deviations: %.3g / %.3g / %.3g"%(
                            self.potential_pool_density_1,
                            self.potential_pool_density_2,
                            self.potential_pool_density_3,))
        else:
            self.dense_connections(input_dimensions, output_dimensions, potential_pool)

        self.num_inputs  = self.sources.shape[1]
        # Random permanence and synapse initialization
        self.permanences = np.random.random(self.sources.shape)
        self.synapses    = self.permanences > self.permanence_threshold
        self.reset()

        if diag:
            # TODO: Synapse Manager's init diag should print the potential pool fraction
            # I'd like to to print: "XXX / YYYY"
            #   where XXX is the size of the potential pool
            #     and YYYY is the maximum size of the potential pool
            #     and XXX/YYYY is the potential fraction.

            if radii is not None:
                print("\tRadii", tuple(radii), '\tNum Inputs', self.num_inputs)
            else:
                print('\tNum Inputs', self.num_inputs)

    def reset(self):
        self.inputs = np.zeros(self.sources.shape, dtype=np.bool)
        # Refresh synapses too, incase it got modified and never updated
        self.synapses = self.permanences > self.permanence_threshold

    def convolution_connections(self, input_dimensions, column_dimensions, radii):
        """
        Sets up the sliding window receptive areas for the spatial pooler
        Directly sets the sources array, no returned value.
        """
        assert(len(column_dimensions) == len(radii))

        # Extended column shape to the input shape
        column_dimensions = tuple(column_dimensions) + (1,)*(len(input_dimensions)-len(column_dimensions))

        # Index offsets into the receptive field, neuron is at center
        window_ranges = [slice(-r, r+1) for r in radii]
        # Broadcast over entirety of extra/non-convolutional dimensions.
        window_ranges += [slice(0, inp_extent) for inp_extent in input_dimensions[len(radii):]]
        # window_index[input-dimension][:] = [coordinates of receptive field]
        window_index = np.mgrid[window_ranges]
        window_index = np.array([dim.flatten() for dim in window_index], dtype=np.float32)

        # Find where the columns are in the input.  (NOTE Columns === Outputs)
        # Assume that they are evenly spaced and that the input space wraps around
        column_ranges = [slice(0, size) for size in column_dimensions]
        # column_locations[input-dimension][:] = vector of locations in the input space, one for each column.
        column_locations = [dim.flatten() for dim in np.mgrid[column_ranges]]
        column_locations *= np.divide(input_dimensions, column_dimensions).reshape(len(input_dimensions), 1)

        # Apply the window offsets to each column location and record the resulting indecies.
        column_locations = column_locations.reshape(column_locations.shape + (1,))
        window_index = window_index.reshape((window_index.shape[0], 1, window_index.shape[1]))
        # index[neuron-index][:] = [input index vector]
        index = column_locations + window_index     # Broadcasting
        # print('window_index', window_index.shape, 'column_locations', column_locations.shape, 'index', index.shape)
        index = np.array(np.rint(index), dtype=np.int64)

        # Collapse the input dimension of the index into a single index into the flattened input.
        index = np.ravel_multi_index(index, input_dimensions, mode='wrap')
        self.sources = index

    def subsample_connections(self, potential_pool):
        """Randomly keep 'potential_pool' fraction of inputs to every output."""
        if potential_pool <= 1:
            num_inputs = int(round(self.sources.shape[1] * potential_pool))
        else:
            num_inputs = potential_pool
        shuffled = np.random.permutation(self.sources.T)
        shuffled = shuffled[:num_inputs, :]
        self.sources = np.sort(shuffled, axis=0).T

    def normally_distributed_connections(self, input_dimensions, column_dimensions, potential_pool, radii):
        """
        Connects each column to its inputs.

        Sets the attribute self.inhibition_radii which is the radii, converted
        into column space units.

        This sets the following attributes:
            potential_pool_density_1
            potential_pool_density_2
            potential_pool_density_3
        Which measure the average fraction of inputs which are potentially
        connected to each column, looking within the first three standard
        deviations of the columns receptive field.  The areas are non-
        overlapping.
        """
        assert(len(column_dimensions) == len(radii))
        radii = np.array(radii)
        input_space_size = np.product(input_dimensions)
        # Clean up the potential_pool parameter.
        if potential_pool <= 1:
            potential_pool = potential_pool * input_space_size
        potential_pool = int(round(potential_pool))

        # Split the input space into topological and extra dimensions.
        topo_dimensions  = input_dimensions[: len(column_dimensions)]
        extra_dimensions = input_dimensions[len(column_dimensions) :]

        num_columns    = int(np.product(column_dimensions))
        self.sources   = np.empty((num_columns, potential_pool), dtype=np.int)

        # Density Statistics
        self.potential_pool_density_1 = 0
        self.potential_pool_density_2 = 0
        self.potential_pool_density_3 = 0
        extra_area   = np.product(extra_dimensions)
        num_inputs_1 = extra_area * math.pi * np.product(radii)
        num_inputs_2 = extra_area * math.pi * np.product(2 * radii)
        num_inputs_3 = extra_area * math.pi * np.product(3 * radii)
        num_inputs_2 -= num_inputs_1
        num_inputs_3 -= num_inputs_1 + num_inputs_2

        # Find where the columns are in the input.  Extra input dimensions are
        # not represented here.
        column_ranges = [slice(0, size) for size in column_dimensions]
        # column_locations[input-dimension][:] = vector locations in input
        # space, one for each column.
        column_locations = [dim.flatten() for dim in np.mgrid[column_ranges]]
        padding   = radii   #
        expand_to = np.subtract(topo_dimensions, np.multiply(2, padding))
        column_spacing    = np.divide(expand_to, column_dimensions).reshape(len(topo_dimensions), 1)
        column_locations *= column_spacing
        column_locations += np.array(padding).reshape(len(topo_dimensions), 1)
        self.inhibition_radii = radii / np.squeeze(column_spacing)

        for column_index in range(num_columns):
            center = column_locations[:, column_index]
            # Make potential-pool many unique input locations.  This is an
            # iterative process: sample the normal distribution, reject
            # duplicates, repeat until done.  Working pool holds the
            # intermediate input-coordinates until it's filled and ready to be
            # spliced into self.sources[column-index, :]
            working_pool  = np.empty((0, len(input_dimensions)), dtype=np.int)
            empty_sources = potential_pool  # How many samples to take.
            for attempt in range(10):
                # Sample points from the input space and cast to valid indecies.
                # Take more samples than are needed B/C some will not be viable.
                topo_pool     = np.random.normal(center, radii, 
                                    size=(max(256, 2*empty_sources), len(radii)))
                topo_pool     = np.rint(topo_pool)   # Round towards center
                out_of_bounds = np.where(np.logical_or(topo_pool < 0, topo_pool >= topo_dimensions))
                topo_pool     = np.delete(topo_pool, out_of_bounds, axis=0)
                extra_pool = np.random.uniform(0, extra_dimensions, size=(topo_pool.shape[0], len(extra_dimensions)))
                extra_pool = np.floor(extra_pool) # Round down to stay in half open range [0, dim)
                # Combine topo & extra dimensions into input space coordinates.
                pool       = np.concatenate([topo_pool, extra_pool], axis=1)
                pool       = np.array(pool, dtype=np.int)
                # Add the points to the working pool
                working_pool = np.concatenate([working_pool, pool], axis=0)
                # Reject duplicates
                working_pool  = np.unique(working_pool, axis=0)
                empty_sources = potential_pool - working_pool.shape[0]
                if empty_sources <= 0:
                    break
            else:
                if empty_sources > .05 * potential_pool:
                    raise ValueError("Not enough sources to fill potential pool.")
                else:
                    print("Warning: Could not find enough unique inputs, allowing %d duplicates..."%empty_sources)
                    duplicates = np.random.randint(0, working_pool.shape[0], size=empty_sources)
                    duplicates = working_pool[duplicates]
                    working_pool = np.concatenate([working_pool, duplicates], axis=0)
            working_pool = working_pool[:potential_pool, :] # Discard extra samples

            # Measure some statistics about input density.
            displacements = working_pool[:, :len(topo_dimensions)] - center
            # Measure in terms of standard deviations of their distribution.
            deviations = displacements / radii
            distances  = np.sum(deviations**2, axis=1)**.5
            pp_size_1  = np.count_nonzero(distances <= 1)
            pp_size_2  = np.count_nonzero(np.logical_and(distances > 1, distances <= 2))
            pp_size_3  = np.count_nonzero(np.logical_and(distances > 2, distances <= 3))
            self.potential_pool_density_1 += pp_size_1 / num_inputs_1
            self.potential_pool_density_2 += pp_size_2 / num_inputs_2
            self.potential_pool_density_3 += pp_size_3 / num_inputs_3

            # Flatten and write to output array.
            working_pool = np.ravel_multi_index(working_pool.T, input_dimensions)
            self.sources[column_index, :] = working_pool
        self.potential_pool_density_1 /= num_columns
        self.potential_pool_density_2 /= num_columns
        self.potential_pool_density_3 /= num_columns

    def dense_connections(self, input_dimensions, output_dimensions, potential_pool):
        """
        Connect every potential_pool inputs to every output.
        Directly sets the sources array, no returned value.
        """
        input_space_size = np.product(input_dimensions)
        input_space      = range(input_space_size)
        if potential_pool <= 1:
            potential_pool = potential_pool * input_space_size
        potential_pool = int(round(potential_pool))
        potential_pool = min(input_space_size, potential_pool)
        num_outputs  = int(np.product(output_dimensions))
        self.sources = np.empty((num_outputs, potential_pool), dtype=np.int)
        for output in range(num_outputs):
            self.sources[output] = np.random.choice(input_space, potential_pool, replace=False)

    def compute(self, input_activity):
        """
        This uses the given presynaptic activity to determine the postsynaptic
        excitment.

        Returns the excitement as a flattened array.  
                Reshape to output_dimensions if needed.
        """
        if isinstance(input_activity, tuple) or input_activity.shape != self.input_dimensions:
            # It's significantly faster to make sparse inputs dense than to use
            # np.in1d, especially since this does NOT discard inactive columns.
            dense = np.zeros(self.input_dimensions, dtype=np.bool)
            dense[input_activity] = True
            input_activity = dense
        else:
            assert(input_activity.dtype == np.bool) # Otherwise self.learn->np.choose breaks

        # Gather the inputs, mask out disconnected synapses, and sum for excitements.

        self.input_activity = input_activity.reshape(-1)
        self.inputs         = self.input_activity[self.sources]

        connected_inputs = np.logical_and(self.synapses, self.inputs)
        excitment        = np.sum(connected_inputs, axis=1)
        return excitment
        return excitment.reshape(self.output_dimensions)

    def compute_sparse(self, input_activity):
        """
        This uses the given presynaptic activity to determine the postsynaptic
        excitment.

        Returns the excitement as a flattened array.  
                Reshape to output_dimensions if needed.
        """
        assert(isinstance(input_activity, tuple))

        num_inputs = np.product(self.input_dimensions)
        input_activity = np.ravel_multi_index(input_activity, self.input_dimensions)

        input_set = set(input_activity)
        vec = np.vectorize(input_set.__contains__, otypes=[np.bool])
        self.inputs = vec(self.sources)

        # Gather the inputs, mask out disconnected synapses, and sum for excitements.
        connected_inputs = np.logical_and(self.synapses, self.inputs)
        excitment        = np.sum(connected_inputs, axis=1)
        return excitment
        return excitment.reshape(self.output_dimensions)

    def learn_outputs(self, output_activity):
        """
        Update permanences and then synapses.

        Argument output_activity is index array
        """
        updates = np.choose(self.inputs[output_activity], 
                            np.array([-self.coincidence_dec, self.coincidence_inc]))
        updates = np.clip(updates + self.permanences[output_activity], 0.0, 1.0)
        self.permanences[output_activity] = updates
        self.synapses[output_activity]    = updates > self.permanence_threshold
    learn = learn_outputs

    def learn_inputs(self, output_activity):
        """
        Update permanences and then synapses.

        Instead of decrementing permanences when then output fires without the
        input, decrement when the input fires and the output doesn't.  This is
        the difference bewtween P->Q and Q->P, or in this context:
            neuron activation -> Proximal/Basal input
              and
            Apical input -> neuron activation

        Argument output_activity is index array
        """
        # Find all inputs which coincided with the output.
        # input_sources.shape = (num-active-outputs, num-input-synapses)
        input_sources = self.sources[output_activity]
        # Input activity is boolen, is the dense input for the current cycle
        reinforce = self.input_activity[input_sources]
        self.permanences[output_activity][reinforce] += self.coincidence_inc

        # Find all inputs which did not coincide with the output.
        output_inactivity = np.ones(self.num_outputs, dtype=np.bool)
        output_inactivity[output_activity] = False
        # These are the input sources for inactive outputs
        inactive_input_sources = self.sources[output_inactivity]
        # Input activity is boolen, is the dense input for the current cycle
        depress = self.input_activity[inactive_input_sources]
        self.permanences[output_inactivity][depress] -= self.coincidence_dec

        self.synapses = self.permanences > self.permanence_threshold

    def synapse_histogram(self, diag=True):
        """Note: diag does NOT show the figure! use plt.show()"""
        data = np.sum(self.synapses, axis=1)
        hist, bins = np.histogram(data, 
                        bins=self.num_inputs, 
                        range=(0, self.num_inputs),
                        density=True)
        if diag:
            import matplotlib.pyplot as plt
            plt.figure(instance_tag + " Synapses")
            plt.plot(bins[:-1], hist*100)
            plt.title("Histogram of Synapse Counts")
            plt.xlabel("Number of connected synapses")
            plt.ylabel("Percent of segments")
            # plt.show()
        return hist, bins

    def permanence_histogram(self, diag=True):
        """Note: diag does NOT show the figure! use plt.show()"""
        data = self.permanences.flatten()
        hist, bins = np.histogram(data, 
                        bins=50, 
                        density=True)
        if diag:
            import matplotlib.pyplot as plt
            plt.figure(instance_tag + " Permanences")
            plt.plot(bins[1:], hist)
            plt.title("Histogram of Permanences")
            plt.xlabel("Permanence")
            plt.ylabel("Percent of segments")
            # plt.show()
        return hist, bins

class SpatialPoolerParameters(Parameters):
    parameters = [
        "column_dimensions",
        "radii",
        "potential_pool",
        "coincidence_inc",
        "coincidence_dec",
        "permanence_thresh",
        "sparsity",
        "boosting_alpha",
    ]
    def __init__(self,
        column_dimensions,
        coincidence_inc     = 0.04,
        coincidence_dec     = 0.01,
        permanence_thresh   = 0.4,
        radii               = None,
        potential_pool      = None,
        sparsity            = 0.02,
        boosting_alpha      = 0.001, ):
        """
        TODO: Copy the docstring from SynapseManager down to here.

        Argument radii is the standard deviation of the gaussian window which
        defines the local neighborhood of a column.  The radii determine which
        inputs are likely to be in a columns potential pool.
        In a normal distribution:
            68% of area is within one standard deviation
            95% of area is within two standard deviations
                  This what's currently done.
            99% of area is within three standard deviations

        Argument boosting_alpha is the small constant used by the moving 
                 exponential average which tracks each columns activation 
                 frequency.
        """
        # Get the parent class to save all these parameters.
        kw_args = locals().copy()
        del kw_args['self']
        del kw_args['__class__']
        super().__init__(**kw_args)

class SpatialPooler(SaveLoad):
    """
    This class handles the mini-column structures and the feed forward 
    proximal inputs to each cortical mini-column.


    This implementation is based on but differs from the one described by
    Numenta's Spatial Pooler white paper, (Cui, Ahmad, Hawkins, 2017, "The HTM
    Spatial Pooler - a neocortical...") in two main ways, the boosting function
    and the local inhibition mechanism.


    Logarithmic Boosting Function:

    This uses a logarithmic boosting function.  Its input is the activation
    frequency which is in the range [0, 1] and its output is a boosting factor
    to multiply each columns excitement by.  It's equation is:
        boost-factor = log( activation-frequency ) / log( target-frequency )
    Some things to note:
        1) The boost factor asymptotically approaches infinity as the activation
           frequency approaches zero.
        2) The boost factor equals zero when the actiavtion frequency of one.
        3) The boost factor for columns which are at the target activation 
           frequency is one.  
        4) This mechanism has a single parameter: boosting_alpha which controls
           the exponential moving average which tracks the activation frequency.


    Fast Local Inhibition:
    This activates the top K most excited columns globally, after normalizing
    all columns by their local area mean and standard deviation.  The local area
    is a gaussian window and the standard deviation of the gaussian is
    proportional to the radius of the receptive field.

    In pseudo code:
    1.  mean_normalized = excitement - gaussian_blur( excitement, radius )
    2.  standard_deviation = sqrt( gaussian_blur( mean_normalized ^ 2, radius ))
    3.  normalized = mean_normalized / standard_deviation
    4.  activate = top_k( normalized, sparsity * number_of_columns )
    """
    data_extension = '.sp'

    stability_st_period = 1000
    stability_lt_period = 10       # Units: self.stability_st_period

    def __init__(self, parameters, input_dimensions, stability_sample_size=0):
        """
        Argument parameters is an instance of SpatialPoolerParameters.

        Argument stability_sample_size, set to 0 to disable stability
                 monitoring, default is off.  
        """
        assert(isinstance(parameters, SpatialPoolerParameters))
        self.args = args           = parameters
        self.input_dimensions      = tuple(input_dimensions)
        self.column_dimensions     = tuple(int(round(cd)) for cd in args.column_dimensions)
        self.num_columns           = np.product(self.column_dimensions)
        self.topology              = args.radii is not None
        self.age                   = 0
        self.stability_schedule    = [0] if stability_sample_size > 0 else [-1]
        self.stability_sample_size = stability_sample_size
        self.stability_samples     = []

        if False:
            self.proximal = SynapseManager(self.input_dimensions, 
                                            self.column_dimensions,
                                            args.radii,
                                            potential_pool=args.potential_pool,
                                            diag=False)
        else:
            self.proximal = htm_cython.SynapseManager_implicit_synapse(
                                            self.input_dimensions, 
                                            self.column_dimensions,
                                            args.radii,
                                            potential_pool=args.potential_pool,
                                            diag=False)
        self.proximal.coincidence_inc    = args.coincidence_inc
        self.proximal.coincidence_dec    = args.coincidence_dec
        self.proximal.permanence_thresh  = args.permanence_thresh
        self.proximal.reset()   # Updates self.proximal.synapses w/ new threshold

        # Initialize to the target activation frequency/sparsity.
        self.average_activations = np.full(self.num_columns, args.sparsity, dtype=np.float)
        self.output = ()

    def compute(self, input_sdr, focus=None, learn=True, diag=False):
        """
        Returns tuple of column indecies

        Note: this methods diags are built for 2D image processing and will not
        work with other types/dimensions of data.  
        """
        args = self.args
        if True:    # Works
            self.zz_raw = raw_excitment = self.proximal.compute(input_sdr)
        else:       # Does not work
            self.zz_raw = raw_excitment = self.proximal.compute_sparse(input_sdr)
            # dense_method = self.proximal.compute(input_sdr)
            # assert(np.all(raw_excitment == dense_method))

        # Logarithmic Boosting Function
        #
        # Disable boosting during evaluations unless topology is a factor.
        #
        # Apply boosting during evaluations if there is a topology because
        # boosting makes the local inhibition algorithm work better.  Boosting
        # corrects for neurons which are systematically under/over-excited.
        # Without boosting the mean excitement in an area is pulled down by
        # outliers, the std-dev increases, the true outliers --correct column
        # activations-- are then divided by too large of a std dev to compete
        # with areas which have more homogenous excitements.  Those areas with
        # more equal excitements tend to be areas with fewer and less
        # interesting inputs available.
        #
        if learn or self.topology:
            boost = np.log2(self.average_activations) / np.log2(args.sparsity)
            boost = np.nan_to_num(boost)
            self.zz_boostd = raw_excitment = boost * raw_excitment

        # Fast Local Inhibition
        if self.topology:
            inhibition_radii    = self.proximal.inhibition_radii
            raw_excitment       = raw_excitment.reshape(self.column_dimensions)
            avg_local_excitment = scipy.ndimage.filters.gaussian_filter(
                                    # Truncate for speed
                                    raw_excitment, inhibition_radii, mode='reflect', truncate=3.0)
            local_excitment     = raw_excitment - avg_local_excitment
            stddev              = np.sqrt(scipy.ndimage.filters.gaussian_filter(
                                    local_excitment**2, inhibition_radii, mode='reflect', truncate=3.0))
            raw_excitment       = np.nan_to_num(local_excitment / stddev)
            self.zz_norm        = raw_excitment.reshape(self.column_dimensions)
            raw_excitment       = raw_excitment.flatten()

        # FOCUS
        #
        # Focus is applied after the local inhibition step. The idea is that
        # this overrides local inhibition. Boosting is applied before local
        # inhibition because (1) it corrects  for systematic over/under
        # excitement, and (2) it's goal is to encourage individual columns to
        # participate more which yields a better representation. However the
        # focus mechanism should be able to boost entire areas of interest.  Its
        # goal is to bias more useful neurons towards activating which helps
        # other areas by filtering out noise.  Blank areas of the image can be
        # considered noise.
        #
        if focus is not None:
            apical_excitement = self.apical.compute(focus)
            mf = self.focus_max
            a  = mf / self.focus_satur
            zz_focus = (1 + np.minimum(mf, a * apical_excitement))
            zz_focused = raw_excitment = raw_excitment * zz_focus

        # Activate the most excited columns.
        #
        k = int(round(self.num_columns * args.sparsity))
        k = max(k, 1)
        self.active_columns = np.argpartition(-raw_excitment, k-1)[:k]

        unflat_output = np.unravel_index(self.active_columns, self.column_dimensions)

        if learn:
            # Update the exponential moving average of each columns activation frequency.
            alpha = args.boosting_alpha
            self.average_activations *= (1 - alpha)                 # Decay with time
            self.average_activations[self.active_columns] += alpha  # Incorperate this sample

            # Update proximal synapses and their permenances.
            self.proximal.learn(self.active_columns)

            if focus is not None:
                self.apical.learn_inputs(self.active_columns)

            self.stability(input_sdr, unflat_output)

            self.age += 1

        if diag:
            # Make sparse input dense
            if isinstance(input_sdr, tuple) or input_sdr.shape != args.input_dimensions:
                dense = np.zeros(args.input_dimensions, dtype=np.bool)
                dense[input_sdr] = True
                input_sdr = dense
            # Make input a valid image, even if that means dropping feature layers.
            if len(input_sdr.shape) == 3 and input_sdr.shape[2] not in (1, 3):
                input_sdr = input_sdr[:,:,0]

            from matplotlib import pyplot as plt
            plt.figure(instance_tag + " SP pipeline (%s)"%random_tag())
            plt.subplot(2, 3, 1)
            plt.imshow(input_sdr, interpolation='nearest')
            plt.title("Input")

            plt.subplot(2, 3, 2)
            if focus is None:
                plt.imshow(self.zz_raw.reshape(self.column_dimensions), interpolation='nearest')
                plt.title('Raw Excitement, radius ' + str(self.args.adii))
            else:
                hist, bins = np.histogram(self.zz_boostd, 
                                bins=self.proximal.num_inputs, 
                                range=(0, self.proximal.num_inputs),
                                density=True)
                fhist, fbins = np.histogram(zz_focused, 
                                bins=self.proximal.num_inputs, 
                                range=(0, self.proximal.num_inputs),
                                density=True)
                plt.plot(bins[:-1], hist*100, 'r', fbins[:-1], fhist*100, 'g')
                plt.title("Histogram of Excitement\nRed = Boosted, Green = Focused")
                plt.xlabel("Postsynaptic Excitement")
                plt.ylabel("Percent of columns")

            plt.subplot(2, 3, 3)
            if learn:
                plt.imshow(self.zz_boostd.reshape(self.column_dimensions),
                            interpolation='nearest')
                plt.title('Boosted (alpha = %g)'%self.average_activations_alpha)
            else:
                plt.imshow(self.average_activations.reshape(self.column_dimensions),
                            interpolation='nearest')
                plt.title('Average Duty Cycle (alpha = %g)'%self.average_activations_alpha)

            if focus is not None:
                plt.subplot(2, 3, 4)
                plt.imshow(zz_focus.reshape(self.column_dimensions), interpolation='nearest')
                plt.title('Apical Excitement')

            if args.topology:
                plt.subplot(2, 3, 5)
                plt.imshow(self.zz_norm, interpolation='nearest')
                plt.title('Locally Inhibited Excitement')

            plt.subplot(2, 3, 6)
            active_state_visual = np.zeros(self.column_dimensions)
            active_state_visual[unflat_output] = 1
            plt.imshow(np.dstack([active_state_visual]*3), interpolation='nearest')
            plt.title("Active Columns (%d train cycles)"%self.age)

            if True:
                # Useful for testing boosting functions.
                self.print_activation_fequency()

            if focus is not None:
                print('Apical boost min ',  np.min(zz_focus) * 100, '%')
                print('Apical boost mean', np.mean(zz_focus) * 100, '%')
                print('Apical boost std ',  np.std(zz_focus) * 100, '%')
                print('Apical boost max ',  np.max(zz_focus) * 100, '%')

        # Keep this around a little while because it's useful.
        # if learn:     # I don't know why this was here.
        self.output = unflat_output

        return unflat_output

    def make_output_dense(self, output=None):
        """
        Returns the output as a dense boolean ndarray.
        If no output is given, uses the most recently computed output.
        """
        if output is None:
            output = np.unravel_index(self.active_columns, self.column_dimensions)
        output = np.array(output)
        dense = np.zeros(self.column_dimensions, dtype=np.bool)
        dense[output] = True
        return dense

    def plot_boost_functions(self, beta = 15):
        # Generate sample points
        dc = np.linspace(0, 1, 10000)
        from matplotlib import pyplot as plt
        fig = plt.figure(1)
        ax = plt.subplot(111)
        log_boost = lambda f: np.log(f) / np.log(self.args.sparsity)
        exp_boost = lambda f: np.exp(beta * (self.args.sparsity - f))
        logs = [log_boost(f) for f in dc]
        exps = [exp_boost(f) for f in dc]
        plt.plot(dc, logs, 'r', dc, exps, 'b')
        plt.title("Boosting Function Comparison \nLogarithmic in Red, Exponential in Blue (beta = %g)"%beta)
        ax.set_xlabel("Activation Frequency")
        ax.set_ylabel("Boost Factor")
        plt.show()

    def entropy(self, diag=True):
        """
        Calculates the entropy of column activations.

        Result is normalized to range [0, 1]
        A value of 1 indicates that all columns are equally and fully utilized.
        """
        p = self.average_activations
        def entropy(p):
            # Binary entroy function
            p_ = (1 - p)
            s = -p*np.log2(p) -p_*np.log2(p_)
            return np.mean(np.nan_to_num(s))
        e = entropy(p) / entropy(self.args.sparsity)
        if diag:
            print("Inst. SP Entropy %g"%e)
        return e

    def stability(self, input_sdr, output_sdr, diag=True):
        """
        Measures the short and long term stability from compute's input stream.
        Do not call this directly!  Instead set it up before and via 
        SpatialPooler.__init__() and this will print the results to STDOUT.

        Argument input_sdr, output_sdr ...

        Attribute stability_sample_size is how many samples to take during each
                  sample period.  

        Attribute stability_samples is list of samples, where each sample is a 
                  list of pairs of (input_sdr, output_sdr).  The index is how 
                  many (short term) sample periods ago the sample was taken.

        Attribute stability_schedule is a list of ages to take input/output
                  samples at, in descending order so that the soonest sample age
                  is at the end of the list.   Append -1 to the schedule to
                  disable stability monitoring. The final age in the schedule is
                  special, on this age it calculates the stability and makes a
                  new schedule for the next period.

        Class Attribute stability_st_period
                st == short term, lt == long term
                The stability period is how many compute cycles this SP will
                wait before recomputing the stability samples and comparing with
                the original results. This calculates two measures of stability:
                short and long term.  The long  term period is written in terms
                of the short term period.

        Class Attribute stability_lt_period
                    Units: self.stability_st_period

        Attribute st_stability, lt_stability are the most recent measurements of
                  short and long term stability, respectively.  These are 
                  initialized to None.
        """
        if self.stability_schedule[-1] != self.age:
            return
        else:
            self.stability_schedule.pop()

        if self.stability_schedule:
            # Not the final scheduled checkup. Take the given sample and return.  
            self.stability_samples[0].append((input_sdr, output_sdr))
            return
        # Else: calculate the stability and setup for the next period of 
        # stability sampling & monitoring.  

        def overlap(a, b):
            a = set(zip(*a))
            b = set(zip(*b))
            overlap = len(a.intersection(b))
            overlap_pct = overlap / min(len(a), len(b))
            return overlap_pct

        # Rerun the samples through the machine.  
        try:
            st_samples = self.stability_samples[1]
        except IndexError:
            self.st_stability = None    # This happens when age < 2 x st_period
        else:
            st_rerun = [self.compute(inp, learn=False) for inp, out in st_samples]
            self.st_stability = np.mean([overlap(re, io[1]) for re, io in zip(st_rerun, st_samples)])

        try:
            lt_samples = self.stability_samples[self.stability_lt_period]
        except IndexError:
            self.lt_stability = None    # This happens when age < st_period X (lt_period + 1)
        else:
            lt_rerun   = [self.compute(inp, learn=False) for inp, out in lt_samples]
            self.lt_stability = np.mean([overlap(re, io[1]) for re, io in zip(lt_rerun, lt_samples)])

        # Make a new sampling schedule.
        sample_period = range(self.age + 1, self.age + self.stability_st_period)
        self.stability_schedule = random.sample(sample_period, self.stability_sample_size)
        # Add the next stability calculation to the end of the schedule.  
        self.stability_schedule.append(sample_period.stop)
        self.stability_schedule.sort(reverse=True)
        # Roll the samples buffer.
        self.stability_samples.insert(0, [])
        self.stability_samples = self.stability_samples[:self.stability_lt_period + 1]

        # Print output
        if diag:
            s = ""
            if self.st_stability is not None:
                s += "Stability (%d) %-5.03g"%(self.stability_st_period, self.st_stability,)
            if self.lt_stability is not None:
                s += " | (x%d) %-5.03g"%(self.stability_lt_period, self.lt_stability)
            if s:
                print(s)

    def noise_perturbation(self, inp, flip_bits, diag=False):
        """
        Measure the change in SDR overlap after moving some of the ON bits.
        """
        tru = self.compute(inp, learn=False)

        # Make sparse input dense.
        if isinstance(inp, tuple) or inp.shape != self.args.input_dimensions:
            dense = np.zeros(self.args.input_dimensions)
            dense[inp] = True
            inp = dense

        # Move some of the on bits around.
        on_bits  = list(zip(*np.nonzero(inp)))
        off_bits = list(zip(*np.nonzero(np.logical_not(inp))))
        flip_bits = min(flip_bits, min(len(on_bits), len(off_bits)) )
        flip_off = random.sample(on_bits, flip_bits)
        flip_on = random.sample(off_bits, flip_bits)
        noisy = np.array(inp, dtype=np.bool)      # Force copy
        noisy[list(zip(*flip_off))] = False
        noisy[list(zip(*flip_on))] = True

        # Calculate the overlap in SP output after adding noise.
        near = self.compute(noisy, learn=False)
        tru  = set(zip(*tru))
        near = set(zip(*near))
        overlap = len(tru.intersection(near))
        overlap_pct = overlap / len(tru)
        if diag:
            print("SP Noise Robustness (%d flipped) %g"%(flip_bits, overlap_pct))
        return overlap_pct

    def noise_robustness(self, inps, diag=False):
        """
        Plot the noise robustness as a function.
        Argument 'inps' is list of encoded inputs.
        """
        if False:
            # Range         Num Samples     Resolution
            # [0, 10)       20              .5
            # [10, 50)      40              1
            # [50, 100]     11              5
            noises = list(np.arange(20) / 2) + list(np.arange(10, 40)) + list(np.arange(11) * 5 + 50)
        elif False:
            # Exponential progression of noises, samples many orders of magnitude of noise.
            num_samples = 50
            x = np.exp(np.arange(num_samples))
            noises = list(x * 100 / np.max(x))
        else:
            # Number of ON bits in encoded input-space +1
            nz = int(round(np.mean([np.count_nonzero(s) for s in inps[:10]])))
            noises = list(np.arange(nz + 1))
            cutoff = len(noises) // 10          # First 'cutoff' many samples have full accuracy.
            while len(noises) > 50 + cutoff:    # Decimate to a sane number of sample points
                noises = noises[:cutoff] + noises[cutoff::2]

        pct_over = []
        for n in noises:
            z = 0
            for inp in inps:
                z += self.noise_perturbation(inp, n, diag=False)
            pct_over.append(z/len(inps))

        if diag:
            from matplotlib import pyplot as plt
            plt.figure(1)
            plt.plot(noises, pct_over)
            plt.title('todo')
            plt.xlabel('todo')
            plt.ylabel('todo')
            plt.show()

        return noises, pct_over

    def activation_fequency(self, diag=True):
        f = self.average_activations.reshape(self.column_dimensions)
        plt.figure(instance_tag + ' Duty Cycles')
        plt.imshow(f, interpolation='nearest')
        plt.title('Average Activation Frequency (alpha = %g)'%self.average_activations_alpha)
        return f

    def print_activation_fequency(self):
        aa = self.average_activations
        boost_min  = np.log2(np.min(aa))  / np.log2(self.args.sparsity)
        boost_mean = np.log2(np.mean(aa)) / np.log2(self.args.sparsity)
        boost_max  = np.log2(np.max(aa))  / np.log2(self.args.sparsity)
        print('duty cycle min  %-.04g'%( np.min(aa)*100), '%', ' log-boost %.04g'%boost_min)
        print('duty cycle mean %-.04g'%(np.mean(aa)*100), '%', ' log-boost %.04g'%boost_mean)
        print('duty cycle std  %-.04g'%( np.std(aa)*100), '%')
        print('duty cycle max  %-.04g'%( np.max(aa)*100), '%', ' log-boost %.04g'%boost_max)

    def __str__(self):
        # TODO: This method is useless but I do want the density printouts.
        st = ["Spatial Pooler Parameters"]
        max_param = max(len(p) for p in self.parameters) + 2
        for p in sorted(self.args.parameters):
            pad = max_param - len(p)
            st.append('\t'+p+' '+ '.'*pad +' '+ str(getattr(self, p, None)))
        p = 'density 1/2/3'
        st.append('\t'+p+'.'*(max_param - len(p)) + "%.3g / %.3g / %.3g"%(
                            self.proximal.potential_pool_density_1,
                            self.proximal.potential_pool_density_2,
                            self.proximal.potential_pool_density_3,))
        return '\n'.join(st)


"""
Outstanding Tasks for T.M.

Nupic has a bunch of rules to determine which segment learns when a column bursts

synapses_per_segment unimplemetned

The TM is going to need continuously running metrics, much like the SP's metrics
* instantaneous anomally, exponential average anomally
* instantaneous input overlap? exponential average input overlap?
"""
class TemporalMemoryParameters(Parameters):
    parameters = [
        'cells_per_column',
        'segments_per_cell',
        # 'synapses_per_segment',       # TODO
        'permanence_inc',
        'permanence_dec',
        'mispredict_dec',
        'permanence_thresh',
        'predictive_threshold',     # Segment excitement threshold for predictions
        'learning_threshold',       # Segment excitement threshold for learning
        'burst_cells_to_connect',   # Fraction of unpredicted input to add to new segments.
    ]

class TemporalMemory:
    """
    This implementation is based on the paper: Hawkins J. and Ahmad S. (2016)
    Why Neurons Have Thousands of Synapses, a Theory of Sequency Memory in
    Neocortex. Frontiers in Neural Circuits 10:23 doi: 10.3389/fncir.2016.00023
    """
    def __init__(self, 
        parameters,
        column_dimensions,):
        """
        Argument parameters is an instance of TemporalMemoryParameters
        Argument column_dimensions ...
        """
        self.args = args         = parameters
        self.column_dimensions   = tuple(int(round(cd)) for cd in column_dimensions)
        self.cells_per_column    = int(round(args.cells_per_column))
        self.segments_per_cell   = int(round(args.segments_per_cell))
        self.num_columns         = np.product(self.column_dimensions)
        # self.num_neurons         = self.cells_per_column * self.num_columns   # UNUSED?
        self.output_shape        = (self.num_columns, self.cells_per_column)
        self.anomaly_alpha       = .1
        self.mean_anomaly        = 0

        # SYNAPSE MANAGER
        self.input_dimensions     = (self.num_columns, self.cells_per_column,)
        self.output_dimensions    = (self.num_columns, self.cells_per_column, self.segments_per_cell,)
        self.num_inputs           = np.product(self.input_dimensions)
        self.num_outputs          = np.product(self.output_dimensions)
        self.permanence_inc       = self.args.permanence_inc
        self.permanence_dec       = self.args.permanence_dec
        self.permanence_threshold = self.args.permanence_thresh

        self.sources     = np.empty((self.num_columns, self.cells_per_column, self.segments_per_cell,), dtype=object)
        self.permanences = np.empty_like(self.sources)
        for idx in np.ndindex(self.sources.shape):
            self.sources[idx]     = np.zeros((0,), dtype=np.int)
            self.permanences[idx] = np.zeros((0,), dtype=np.float)
        self.reset()

    def reset(self):
        # Zero the input space
        self.inputs = np.empty_like(self.sources)
        for idx in np.ndindex(self.sources.shape):
            self.inputs[idx] = np.zeros(self.sources[idx].shape, dtype=np.bool)

        # Refresh synapses too, incase it got modified and never updated
        threshold = self.permanence_threshold
        self.synapses = np.empty_like(self.sources)
        for idx in np.ndindex(self.sources.shape):
            self.synapses[idx] = self.permanences[idx] > threshold

        self.excitement  = np.zeros((self.num_columns, self.cells_per_column, self.segments_per_cell), dtype=np.int)
        self.active      = np.empty((2, 0), dtype=np.int)
        self.predictions = np.zeros((self.num_columns, self.cells_per_column), dtype=np.bool)

    def compute(self, column_activations):
        """
        Argument column_activations is an index of active columns.

        Returns active neurons as pair (flat-column-index, neuron-index)

        This sets the attributes anomaly and mean_anomaly.
        """
        ########################################################################
        # PHASE 1:  Determine the currently active neurons.
        ########################################################################
        # Flatten the input column indecies
        columns = np.ravel_multi_index(column_activations, self.column_dimensions)

        # Activate all neurons which are in a predictive state and in an active
        # column.
        active_dense      = self.predictions[columns]
        col_num, neur_idx = np.nonzero(active_dense)
        # This gets the actual column index, undoes the effect of discarding the
        # inactive columns before the nonzero operation.  
        col_idx = columns[col_num]
        active  = (col_idx, neur_idx)

        # If a column activates but was not predicted by any neuron segment,
        # then it bursts.  The bursting columns are the unpredicted columns.
        bursting_columns = np.setdiff1d(columns, col_idx)
        # All neurons in bursting columns activate.
        burst_col_idx  = np.repeat(bursting_columns, self.cells_per_column)
        burst_neur_idx = np.tile(np.arange(self.cells_per_column), len(bursting_columns))
        burst_active   = (burst_col_idx, burst_neur_idx)
        active         = np.concatenate([active, burst_active], axis=1)

        # Anomaly metric
        self.anomaly = len(bursting_columns)/len(columns)
        a = self.anomaly_alpha
        self.mean_anomaly = (1-a)*self.mean_anomaly + a*self.anomaly

        ########################################################################
        # PHASE 2:  Learn about the previous to current timestep transition.
        ########################################################################

        # All segments which meet the learning threshold will learn.
        learning_segments = self.excitement > self.args.learning_threshold

        # All mispredicted segments receive a small permanence penalty (active
        # segments in inactive columns)
        inactive_columns = np.ones(self.num_columns, dtype=np.bool)
        inactive_columns[columns] = False
        mispredictions = learning_segments[inactive_columns]
        # misprediction_index = np.ravel_multi_index(np.nonzero(mispredictions),
        #     (len(inactive_columns), self.cells_per_column, self.segments_per_cell))
        self.permanences[inactive_columns][mispredictions] -= self.args.mispredict_dec

        # Reinforce all segments which correctly predicted an active column.  
        col_num, neur_idx, seg_idx = np.nonzero(learning_segments[columns])
        col_idx   = columns[col_num]
        reinforce = (col_idx, neur_idx, seg_idx)

        if len(bursting_columns):
            # The most excited segment in each bursting column should learn to
            # recognise & predict this state.  First slice out just the bursting
            # columns.  Then flatten the neurons and segments together into 1
            # dimension for argmax to work with.
            bursting_excitement = self.excitement[bursting_columns]
            bursting_excitement = bursting_excitement.reshape(len(bursting_columns), -1)
            bursting_flat_idx   = np.argmax(bursting_excitement, axis=1)
            # Then unflatten the indecies to shape (neuron-index, segment-index)
            bursting_neurons, bursting_segments = np.unravel_index(bursting_flat_idx, 
                                (self.cells_per_column, self.segments_per_cell))
            # Append most excited segments in bursting columns to the reinforce list.
            reinforce_segments = (bursting_columns, bursting_neurons, bursting_segments)
            reinforce = np.concatenate([reinforce, reinforce_segments], axis=1)

        # Do this before adding any synapses
        self.learn_outputs(reinforce)

        if len(bursting_columns):
            # Add synapses to the potential pool from a random subset of
            # the previous timesteps activity to the bursting segments.
            self.make_synapses(self.active, reinforce_segments, .25)

        ########################################################################
        # PHASE 3:  Compute predictions for the next timestep and return.
        ########################################################################

        # Compute the predictions based on the current timestep.
        self.excitement = self.compute_excitment( active )
        self.excitement = self.excitement.reshape(( self.num_columns, 
                                                    self.cells_per_column, 
                                                    self.segments_per_cell))

        self.predictions = np.any(self.excitement > self.args.predictive_threshold, axis=2)

        self.active = tuple(active)    # Save this for the next timestep.
        return self.predictions

    def compute_excitment(self, input_activity):
        """
        This uses the given presynaptic activity to determine the postsynaptic
        excitment.

        Returns the excitement as a flattened array.  
                Reshape to output_dimensions if needed.
        """
        if isinstance(input_activity, tuple) or input_activity.shape != self.input_dimensions:
            # It's significantly faster to make sparse inputs dense than to use
            # np.in1d, especially since this does NOT discard inactive columns.
            dense = np.zeros(self.input_dimensions, dtype=np.bool)
            dense[input_activity] = True
            input_activity = dense
        assert(input_activity.dtype == np.bool) # Otherwise self.learn->np.choose breaks

        # Gather the inputs, mask out disconnected synapses, and sum for excitements.
        self.input_activity = input_activity.reshape(-1)
        excitments = []
        for out in np.ndindex(self.sources.shape):
            assert(self.sources[out].dtype == np.int)
            inputs = np.take(self.input_activity, self.sources[out])
            self.inputs[out] = inputs
            connected_inputs = np.logical_and(self.synapses[out], inputs)
            x = np.sum(connected_inputs)
            excitments.append(x)
        return np.array(excitments)

    def learn_outputs(self, output_activity):
        """
        Update permanences and then synapses.

        Argument output_activity is index array
        """
        for out in zip(*output_activity):
            if not len(self.inputs[out]):
                continue    # Segment is empty...
            updates = np.choose(self.inputs[out], 
                                np.array([-self.permanence_dec, self.permanence_inc]))
            updates = np.clip(updates + self.permanences[out], 0.0, 1.0)
            self.permanences[out] = updates
            self.synapses[out]    = updates > self.permanence_threshold

    def make_synapses(self, input_set, output_index, percent_connected):
        """
        Attempts to add new synapses to the potential pool.

        Argument input_set is index of previously active cells
        Argument output_index is index of winning segments which need more synapses
        Argument percent_connected is the maximum fraction of input_set which 
                is connected to each output in output_set.  The opposite is also
                true: it is the max fraction of output_set which each input 
                connected to.
        """
        # if not len(input_set) or (len(input_set.shape) > 1 and not len(input_set[0])):
        #     return

        # Make input sparse
        if isinstance(input_set, np.ndarray) and input_set.shape == self.input_dimensions:
            input_set = np.nonzero(input_set)

        # Flatten the input and output sets
        inp = np.ravel_multi_index(input_set, self.input_dimensions)
        if not len(inp):
            return
        # out = np.ravel_multi_index(output_index, self.output_dimensions)

        num_connections = int(round(percent_connected * self.num_inputs))
        for site in zip(*output_index):
            # Randomly select inputs which are not already connected to
            existing_connections = set(self.sources[site])
            new_sources = []
            for new_con in range(num_connections):
                for retry in range(10):    # Don't die friends
                    source = random.choice(inp)
                    if source not in existing_connections:
                        new_sources.append(source)
                        existing_connections.add(source)
                        break
                else:
                    # Could not find an unconnected input
                    # print("DEBUG, no unused inputs remaining")
                    break
            # Add the new potential synapses
            self.sources[site]     = np.concatenate([self.sources[site], np.array(new_sources, dtype=np.int)])
            new_permanences        = np.random.random(len(new_sources))
            self.permanences[site] = np.concatenate([self.permanences[site], new_permanences])
            new_synapses           = new_permanences > self.permanence_threshold
            self.synapses[site]    = np.concatenate([self.synapses[site], new_synapses])

    def excitement_histogram(self, diag=True):
        """
        FIXME THIS NEEDS TO BE TIME AVERAGED!!!

        Note: diag does NOT show the figure! use plt.show()
        """
        data = self.excitement
        hist, bins = np.histogram(data, 
                        bins=self.basal.num_inputs, 
                        range=(0, self.basal.num_inputs),
                        density=True)
        if diag:
            import matplotlib.pyplot as plt
            plt.figure(instance_tag + " Excitement")
            max_bin = np.max(np.nonzero(hist)) + 2      # Don't show all the trailing zero
            plt.plot(bins[:max_bin], hist[:max_bin]*100)
            plt.title("Histogram of Excitement\n"+
                "Red line is predictive threshold")
            plt.xlabel("Postsynaptic Excitement")
            plt.ylabel("Percent of segments")
            plt.axvline(self.predictive_threshold, color='r')
            # plt.show()
        return hist, bins


class SDRC_Parameters(Parameters):
    parameters = ['alpha',]
    def __init__(self, alpha=1/1000):
        self.alpha = alpha

class SDR_Classifier(SaveLoad):
    """Maximum Likelyhood classifier for SDRs."""
    data_extension = '.sdrc'
    def __init__(self, parameters, input_shape, output_shape, output_type):
        """
        Argument parameters must be an instance of SDRC_Parameters.
        Argument output_type must be one of: 'index', 'bool', 'pdf'
        """
        self.args         = parameters
        self.input_shape  = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.output_type  = output_type
        assert(self.output_type in ('index', 'bool', 'pdf'))
        # Don't initialize to zero, touch every input+output pair once or twice.
        self.stats = (np.random.random(self.input_shape + self.output_shape) + 1) * self.args.alpha
        self.age = 0

    def train(self, inp, out):
        """
        Argument inp is tuple of index arrays, as output from SP's or TP's compute method
        inp = (ndarray of input space dim 0 indexes, ndarray of input space dim 1 indexes, ...)
        """
        alpha = self.args.alpha
        self.stats[inp] *= (1-alpha)   # Decay
        if self.output_type == 'index':
            try:
                for out_idx in zip(*out):
                    self.stats[inp + out_idx] += alpha
            except TypeError:
                self.stats[inp + out] += alpha

        if self.output_type == 'bool':
            self.stats[inp, out] += alpha

        if self.output_type == 'pdf':
            updates = (out - self.stats[inp]) * alpha
            self.stats[inp] += updates
        self.age += 1

    def predict(self, inp):
        """
        Argument inputs is ndarray of indexes into the input space.
        Returns probability of each catagory in output space.
        """
        # Colapse inputs to a single dimension
        pdf = self.stats[inp].reshape(-1, *self.output_shape)
        if True:
            # Combine multiple probabilities into single pdf. Product, not
            # summation, to combine probabilities of independant events. The
            # problem with this is if a few unexpected bits turn on it
            # mutliplies the result by zero, and the test dataset is going to
            # have unexpected things in it.  
            return np.product(pdf, axis=0, keepdims=False)
        else:
            # Use summation B/C it works well.
            return np.sum(pdf, axis=0, keepdims=False)

    def __str__(self):
        s = "SDR Classifier alpha %g\n"%self.args.alpha
        s += "\tInput -> Output shapes are", self.input_shape, '->', self.output_shape
        return s


class RandomOutputClassifier:
    """
    This classifier uses the frequency of the trained target outputs to generate
    random predictions.  It is used to get a baseline  performance to compare
    against the SDR_Classifier.
    """
    def __init__(self, output_shape):
        self.output_shape = tuple(output_shape)
        self.stats = np.zeros(self.output_shape)
        self.age = 0

    def train(self, out):
        """
        Argument out is tuple of index arrays, SDR encoded value of target output
                     Or it can be a dense boolean array too.
        """
        if True:
            # Probability density functions
            self.stats += out / np.sum(out)
        else:
            # Index or mask arrays
            self.stats[out] += 1
        self.age += 1

    def predict(self):
        """
        Argument inputs is ndarray of indexes into the input space.
        Returns probability of each catagory in output space.
        """
        return self.stats
        return np.random.random(self.output_shape) < self.stats / self.age

    def __str__(self):
        return "Random Output Classifier, Output shape is %s"%str(self.output_shape)
