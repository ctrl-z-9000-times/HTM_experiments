# Written by David McDougall, 2017

import numpy as np
import math
import scipy.ndimage
import random
import cv2
import PIL, PIL.ImageDraw
import matplotlib.pyplot as plt
from genetics import Parameters
from sdr import SDR


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
        assert(isinstance(parameters, RandomDistributedScalarEncoderParameters))
        self.args = args    = parameters
        self.output         = SDR((args.size,))
        self.on_bits        = int(round(self.output.size * args.sparsity))

    def encode(self, value):
        # This must be integer division! Everything under the resolution must be removed.
        index = value // self.args.resolution
        code = np.zeros(self.output.dimensions, dtype=np.bool)
        size = self.output.size
        for offset in range(self.on_bits):
            # Cast to string before hash, python3 will not hash an integer, uses
            # value instead.
            h = hash(str(index + offset))
            bucket = h % size
            # If this bucket is already full, walk around until it finds one
            # that isn't taken.
            while code[bucket]:
                bucket = (bucket + 1) % size
            code[bucket] = True
        self.output.dense = code
        return self.output


class EnumEncoder:
    """
    Encodes arbirary enumerated values.
    There is no semantic similarity between encoded values.
    """
    def __init__(self, bits, sparsity, diag=True):
        self.bits         = bits
        self.sparsity     = sparsity
        self.on_bits      = int(round(self.bits * self.sparsity))
        self.enums        = set()
        self.output_sdr   = SDR((self.bits,))
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
        else:
            1/0
        bits = np.zeros((self.bits,), dtype=np.bool)
        bits_per_enum = int(round(self.bits * self.sparsity / len(names)))
        total_bits = len(names) * bits_per_enum
        for nm in names:
            assert(nm in self.enums)
            r = random.Random(hash(nm))
            b = r.sample(range(self.bits), total_bits)
            b = random.sample(b, bits_per_enum)
            bits[b] = True
        self.output_sdr.dense = bits
        return self.output_sdr


class BWImageEncoder:
    def __init__(self, input_space, diag=True):
        self.output = SDR(tuple(input_space) + (2,))

    def encode(self, image):
        mean = np.mean(image)
        on_bits  = image >= mean
        off_bits = np.logical_not(on_bits)
        self.output.dense = np.dstack([on_bits, off_bits])
        return self.output


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
        # Each bit responds to a range of input values, length of range is 2*Radius.
        radius            = self.len_drange * args.sparsity / 2
        if self.wrap:
            # If wrapping is enabled then don't generate ranges which will be
            # truncated near the edges.
            centers = np.random.uniform(min(self.drange) + radius,
                                        max(self.drange) - radius,
                                        size=self.output_shape)
        else:
            # Ranges within a radius of the edges are OK.  They will not respond
            # to a full range of input values but are needed to represent the
            # bits at the edges of the data range.
            centers = np.random.uniform(min(self.drange),
                                        max(self.drange),
                                        size=self.output_shape)
        # Make the lower and upper bounds of the ranges.
        self.low  = np.array(centers - radius, dtype=self.dtype)
        self.high = np.array(centers + radius, dtype=self.dtype)

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


# EXPERIMENT: Try breaking out each output encoder by type instead of
# concatenating them all together.  Each type of sensors would then get its own
# HTM.  Maybe keep the derivatives with their source?
#
# EXPERIMENT: Add a motion sensor to the EyeSensor.
#
# TODO: Prefix each parameter name with which part of the eye its describing.
#       This would make the print outs much more readable.
#       Example:  'cv_', 'm_', 'v_'
#
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
        'num_cv',
        'pos_stddev',
        'angle_stddev',
        'scale_stddev',
        # Motor Sensor Parameters
        'position_encoder',
        'velocity_encoder',
        'angle_encoder',
        'angular_velocity_encoder',
        'scale_encoder',
        'scale_velocity_encoder',
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
        num_cv       = 600,
        pos_stddev   = 1,
        angle_stddev = math.pi / 8,
        scale_stddev = 2,
        # Motor Sensor Defaults
        position_encoder         = None,
        velocity_encoder         = None,
        angle_encoder            = None,
        angular_velocity_encoder = None,
        scale_encoder            = None,
        scale_velocity_encoder   = None,):
        """
        Argument eye_dimensions ...
        Arguments fovea_param_1 and fovea_param_2 ...
        Arguments min_scale and max_scale ...
        Arguments hue_encoder, sat_encoder and val_encoder are instances of
                  ChannelEncoderParameters.
        Argument edge_encoder is an instance of ChannelThresholderParameters.

        Argument num_cv is the approximate number of control vectors to use.
        Arguments pos_stddev, angle_stddev, and scale_stddev are the standard
                  deviations of the control vector movements, control vectors
                  are normally distributed about a mean of 0.

        Arguments position_encoder, velocity_encoder, angle_encoder, angular_velocity_encoder,
                  scale_encoder, and scale_velocity_encoder are instances of 
                  RandomDistributedScalarEncoderParameters.
        """
        # Get the parent class to save all these parameters.
        kw_args = locals().copy()
        del kw_args['self']
        # Double underscores are magic and come and go as they please.  Filter them all out.
        dunder  = lambda name: name.startswith('__') and name.endswith('__')
        kw_args = {k:v for k,v in kw_args.items() if not dunder(k)}
        super().__init__(**kw_args)
        if hue_encoder is None:
            self.hue_encoder     = ChannelEncoderParameters()
        if sat_encoder is None:
            self.sat_encoder     = ChannelEncoderParameters()
        if val_encoder is None:
            self.val_encoder     = ChannelEncoderParameters()
        if edge_encoder is None:
            self.edge_encoder    = ChannelThresholderParameters()

        # Motor Sensor Defaults
        if self.position_encoder is None:
            self.position_encoder        = RandomDistributedScalarEncoderParameters()
        if self.velocity_encoder is None:
            self.velocity_encoder        = RandomDistributedScalarEncoderParameters()
        if self.angle_encoder is None:
            self.angle_encoder           = RandomDistributedScalarEncoderParameters(
                                            resolution  = math.pi / 80,)
        if self.angular_velocity_encoder is None:
            self.angular_velocity_encoder = RandomDistributedScalarEncoderParameters(
                                            resolution  = math.pi / 80,)
        if self.scale_encoder is None:
            self.scale_encoder           = RandomDistributedScalarEncoderParameters(
                                            resolution  = .2,)
        if self.scale_velocity_encoder is None:
            self.scale_velocity_encoder  = RandomDistributedScalarEncoderParameters(
                                            resolution  = .2,)

# TODO: Should preprocess_edges find the edges before or after casting to
# greyscale?  Currently this finds edges after casting to greyscale so it really
# is only finding the edges in the value channel.  If instead I found the edges
# in all RGB channels and then averaged their magnitudes if could detect edges
# in Hue, Saturation and Value channels.  How would it handle the edge angles?
# Maybe just take the angle from the RGB channel with the greatest magnitude?
class EyeSensor:
    """
    Eye sensor with controllable movement and central fovae.

    This eye sensor has the following design criteria:
    1) The central fovae should be capable of identifying objects in 1 or 2 saccades.
    2) The periferal vision should be capable of locating and tracking objects.

    This eye has 4 degrees of freedom: X and Y location, scale, and orientation.
    These values can be controlled by activating control vectors, each of which 
    has a small but cumulative effect.  CV's are normally distributed with a
    mean of zero.  

    The eye outputs the its current location, scale and orientation as well as 
    their first derivatives w/r/t time as a dense SDR.

    Fun Fact 1: The human optic nerve has 800,000 ~ 1,700,000 nerve fibers.
    Fun Fact 2: The human eye can distiguish between 10 million different colors.
    Sources: Wikipedia.
    """
    def __init__(self, parameters):
        """
        Attribute optic_sdr ... retina's output
        Attribute control_sdr ... eye movement input controls
        Attribute motor_sdr ... internal motor sensor output
        
        Attribute rgb ... The most recent view, kept as a attribute for making diagnostics.

        Attribute position     (X, Y) coords of eye within image, Read/Writable
        Attribute orientation  ... units are radians, Read/Writable
        Attribute scale        ... Read/Writable

        Attribute gaze is a list of tuples of (X, Y, Orientation, Scale)
                    History of recent movements, self.move() updates this.
                    This is cleared by the following methods:
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
        self.optic_sdr = SDR(self.eye_dimensions + (depth,))

        self.control_vectors, self.control_sdr = self.make_control_vectors(
                num_cv       = args.num_cv,
                pos_stddev   = args.pos_stddev,
                angle_stddev = args.angle_stddev,
                scale_stddev = args.scale_stddev,)

        self.motor_position_encoder         = RandomDistributedScalarEncoder(args.position_encoder)
        self.motor_angle_encoder            = RandomDistributedScalarEncoder(args.angle_encoder)
        self.motor_scale_encoder            = RandomDistributedScalarEncoder(args.scale_encoder)
        self.motor_velocity_encoder         = RandomDistributedScalarEncoder(args.velocity_encoder)
        self.motor_angular_velocity_encoder = RandomDistributedScalarEncoder(args.angular_velocity_encoder)
        self.motor_scale_velocity_encoder   = RandomDistributedScalarEncoder(args.scale_velocity_encoder)
        self.motor_encoders = [ self.motor_position_encoder,    # X Posititon
                                self.motor_position_encoder,    # Y Position
                                self.motor_angle_encoder,
                                self.motor_scale_encoder,
                                self.motor_velocity_encoder,    # X Velocity
                                self.motor_velocity_encoder,    # Y Velocity
                                self.motor_angular_velocity_encoder,
                                self.motor_scale_velocity_encoder,]
        self.motor_sdr = SDR((sum(enc.output.size for enc in self.motor_encoders),))

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
    def make_control_vectors(num_cv, pos_stddev, angle_stddev, scale_stddev):
        """
        Argument num_cv is the approximate number of control vectors to create
        Arguments pos_stddev, angle_stddev, and scale_stddev are the standard
                  deviations of the controls effects of position, angle, and 
                  scale.

        Returns pair of control_vectors, control_sdr

        The control_vectors determines what happens for each output. Each
        control is a 4-tuple of (X, Y, Angle, Scale) movements. To move,
        active controls are summed and applied to the current location.
        control_sdr contains the shape of the control_vectors.
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
        return control_vectors, SDR(control_shape)

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
        denom   = 3 * 255.
        grey    = np.sum(self.image/denom, axis=2, keepdims=False, dtype=np.float32)
        sobel_x = scipy.ndimage.sobel(grey, axis=0)
        sobel_y = scipy.ndimage.sobel(grey, axis=1)
        self.edge_angles    = np.arctan2(sobel_y, sobel_x)  # Counterclockwise
        self.edge_magnitues = (sobel_x ** 2 + sobel_y ** 2) ** .5
        assert(self.edge_angles.dtype == np.float32)
        assert(self.edge_magnitues.dtype == np.float32)
        if False:
            plt.figure("EDGES")
            plt.subplot(1,2,1)
            plt.imshow(self.edge_magnitues, interpolation='nearest')
            plt.title("MAG")
            plt.subplot(1,2,2)
            plt.imshow(self.edge_angles, interpolation='nearest')
            plt.title("ANG")
            plt.show()

    def randomize_view(self):
        """Set the eye's view point to a random location"""
        self.orientation = random.random() * 2 * math.pi
        self.scale       = random.uniform(self.args.min_scale, self.args.max_scale)
        eye_radius       = np.multiply(self.scale / 2, self.eye_dimensions)
        self.position    = [np.random.uniform(0, dim) for dim in self.image.shape[:2]]
        # Discard any prior gaze tracking after forcibly moving eye position to new starting position.
        self.gaze = [tuple(self.position) + (self.orientation, self.scale)]
        # Update motor sensors.
        self.control_sdr.zero()     # This sets motor velocity to zero.
        self.move()

    def center_view(self):
        """Center the view over the image"""
        self.orientation = 0
        self.position    = np.divide(self.image.shape[:2], 2)
        self.scale       = np.max(np.divide(self.image.shape[:2], self.eye_dimensions))
        # Discard any prior gaze tracking after forcibly moving eye position to new starting position.
        self.gaze = [tuple(self.position) + (self.orientation, self.scale)]
        # Update motor sensors.
        self.control_sdr.zero()     # This sets motor velocity to zero.
        self.move()

    def move(self, control_sdr=None):
        """
        Apply the given controls to the current gaze location and updates the
        motor sdr accordingly.

        Argument control_sdr is assigned into this classes attribute
                 self.control_sdr.  It represents the control vectors to use.
                 The selected control vectors are summed and their effect is
                 applied to the eye's location.

        Returns an SDR encoded representation of the eyes new location and 
        velocity.
        """
        self.control_sdr.assign(control_sdr)
        # Calculate the forces on the motor
        controls = self.control_vectors[self.control_sdr.index]
        controls = np.sum(controls, axis=0)
        dx, dy, dangle, dscale = controls
        # Calculate the new rotation
        self.orientation = (self.orientation + dangle) % (2*math.pi)
        # Calculate the new scale
        new_scale  = np.clip(self.scale + dscale, self.args.min_scale, self.args.max_scale)
        real_ds    = new_scale - self.scale
        avg_scale  = (new_scale + self.scale) / 2
        self.scale = new_scale
        # Scale the movement such that the same CV yields the same visual
        # displacement, regardless of scale.
        dx       *= avg_scale
        dy       *= avg_scale
        # Calculate the new position.  
        x, y     = self.position
        p        = [x + dx, y + dy]
        p        = np.clip(p, [0,0], self.image.shape[:2])
        real_dp  = np.subtract(p, self.position)
        self.position = p
        # Book keeping.
        self.gaze.append(tuple(self.position) + (self.orientation, self.scale))
        # Put together information about the motor.
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
        # Encode the motors sensors and concatenate them into one big SDR.
        v_enc = [enc.encode(v) for v, enc in zip(velocity, self.motor_encoders)]
        self.motor_sdr.dense = np.concatenate([sdr.dense for sdr in v_enc])
        return self.motor_sdr

    def view(self):
        """
        Returns the image which the eye is currently seeing.

        Attribute self.rgb is set to the current image which the eye is seeing.
        """
        # Rotate the samples points
        c   = math.cos(self.orientation)
        s   = math.sin(self.orientation)
        rot = np.array([[c, -s], [s, c]])   # XY plane counterclockwise
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
                                            order=1)            # Linear interp
            channels.append(ch.reshape(self.eye_dimensions))
        self.rgb = rgb = np.dstack(channels)

        # Convert view to HSV and encode HSV to SDR.
        hsv         = np.array(rgb, dtype=np.float32)
        hsv         /= 255.
        # Performance Note: OpenCV2's cvtColor() is about 40x faster than
        # matplotlib.colors.rgb_to_hsv().
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
                                        order=1)            # Linear interp
        # Both the eye's orientation and the edge directions are measured
        # counterclockwise so subtracting them makes the resulting edge features
        # invariant with respect to relative angle between the eye and the
        # feature.
        angles   -= self.orientation    # Edge encoder does modulus for me.
        angles   = angles.reshape(self.eye_dimensions)
        mags     = mags.reshape(self.eye_dimensions)
        edge_sdr = self.edge_encoder.encode(angles, mags)

        self.optic_sdr.dense = np.dstack([hue_sdr, sat_sdr, val_sdr, edge_sdr])
        return self.optic_sdr

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
                draw.line((y1, x1, y2, x2), fill='black', width=5)
                draw.line((y1, x1, y2, x2), fill='red', width=2)
            # Draw the bounding box of the eye sensor around each gaze point
            for x, y, orientation, scale in self.gaze:
                # Find the four corners of the eye's window
                corners = []
                for ec_x, ec_y in [(0,0), (0,-1), (-1,-1), (-1,0)]:
                    corners.append(self.eye_coords[:, ec_x, ec_y])
                # Convert from list of pairs to index array.
                corners = np.transpose(corners)
                # Rotate the corners
                c = math.cos(orientation)
                s = math.sin(orientation)
                rot = np.array([[c, -s], [s, c]])
                corners = np.matmul(rot, corners)
                # Scale/zoom the corners
                corners *= scale
                # Position the corners
                corners += np.array([x, y]).reshape(2, 1)
                # Convert from index array to list of coordinates pairs
                corners = list(tuple(coord) for coord in np.transpose(corners))
                # Draw the points
                for start, end in zip(corners, corners[1:] + [corners[0]]):
                    line_coords = (start[1], start[0], end[1], end[0],)
                    draw.line(line_coords, fill='green', width=2)
            del draw
            plt.figure("Gaze Tracking")
            im = np.array(im)
            plt.imshow(im, interpolation='nearest')
            plt.show()
        return self.gaze[:]

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
        self.sensor       = sensor = eyesensor
        self.sensor_view  = sensor.view
        self.sensor.view  = self.view
        self.age          = 0
        self.samples      = []
        number_of_samples = min(number_of_samples, sample_period)   # Don't die.
        self.schedule     = random.sample(range(sample_period), number_of_samples)
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
            plt.subplot(rows, cols, idx+1)
            plt.imshow(img, interpolation='nearest')
        plt.show()
