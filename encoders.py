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

    This encoder associates a name with an SDR.  It works by hashing the name,
    seeding a pseudo-random number generator with the hash, and activating a
    random sample of the bits in the output SDR.
    """
    def __init__(self, bits, sparsity, diag=True):
        self.bits         = int(round(bits))
        self.sparsity     = sparsity
        self.on_bits      = int(round(self.bits * self.sparsity))
        self.output_sdr   = SDR((self.bits,))
        if diag:
            print("Enum Encoder: %d bits %.2g%% sparsity"%(bits, 100*sparsity))

    def encode(self, name):
        """
        Accepts a string.

        Returns dense boolean array.
        """
        num_active = int(round(self.bits * self.sparsity))
        enum_rng   = random.Random(hash(name))
        active     = enum_rng.sample(range(self.bits), num_active)
        self.output_sdr.flat_index = np.array(active)
        return self.output_sdr


class ChannelEncoder:
    """
    This assigns a random range to each bit of the output SDR.  Each bit becomes
    active if its corresponding input falls in its range.  By using random
    ranges, each bit represents a different thing even if it mostly overlaps
    with other comparable bits.  This way redundant bits add meaning.
    """
    def __init__(self, input_shape, num_samples, sparsity,
        dtype       = np.float64,
        drange      = range(0,1),
        wrap        = False):
        """
        Argument input_shape is tuple of dimensions for each input frame.

        Argument num_samples is number of bits in the output SDR which will
                 represent each input number, this is the added data depth.

        Argument sparsity is fraction of output which on average will be active.
                 This is also the fraction of the input spaces which (on 
                 average) each bin covers.

        Argument dtype is numpy data type of channel.

        Argument drange is a range object or a pair of values representing the 
                 range of possible channel values.

        Argument wrap ... default is False.
                 This supports modular input spaces and ranges which wrap
                 around. It does this by rotating the inputs by a constant
                 random amount which hides where the discontinuity in ranges is.
                 No ranges actually wrap around the input space.
        """
        self.input_shape  = tuple(input_shape)
        self.num_samples  = int(round(num_samples))
        self.sparsity     = sparsity
        self.output_shape = self.input_shape + (self.num_samples,)
        self.dtype        = dtype
        self.drange       = drange
        self.len_drange   = max(drange) - min(drange)
        self.wrap         = bool(wrap)
        if self.wrap:
            self.offsets  = np.random.uniform(0, self.len_drange, self.input_shape)
            self.offsets  = np.array(self.offsets, dtype=self.dtype)
        # Each bit responds to a range of input values, length of range is 2*Radius.
        radius            = self.len_drange * self.sparsity / 2
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
        """Returns a dense boolean np.ndarray."""
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
        lines.append("\tSparsity %.03g, dtype %s, drange %s %s"%(
                self.sparsity,
                self.dtype.__name__,
                self.drange,
                'Wrapped' if self.wrap else ''))
        return '\n'.join(lines)


class ChannelThresholderParameters(Parameters):
    parameters = [
        'num_samples',
        'sparsity',
        'mean',
        'stddev',]
    def __init__(self, **kw_args):
        """
        Argument num_samples ... see ChannelEncoder
        Argument sparsity ... see ChannelEncoder
        Argument mean is the average of activation thresholds.
        Argument stddev is the standard deviation of activation thresholds.
        """
        super().__init__(**kw_args)

class ChannelThresholder:
    """
    Creates a channel encoder with an additional activation threshold.  A bit
    becomes active if and only if the underlying channel encoder activates it
    and its magnitude is not less than its threshold. Activation thresholds are
    normally distributed.
    """
    def __init__(self, parameters, input_shape,  dtype, drange, wrap):
        """
        Argument parameters is an instance of ChannelThresholderParameters.
        Argument input_shape is tuple of dimensions of each input frame.
        Arguments dtype, drange, and wrap are passed through to the underlying
                  channel encoder.
        """
        assert(isinstance(parameters, ChannelThresholderParameters))
        self.args = args  = parameters
        self.channel      = ChannelEncoder(input_shape, args.num_samples, args.sparsity,
                            dtype=dtype, drange=drange, wrap=wrap)
        self.output_shape = self.channel.output_shape
        self.thresholds   = np.random.normal(args.mean, args.stddev, self.output_shape)
        self.thresholds   = np.array(self.thresholds, dtype)

    def encode(self, img_data, magnitude):
        """
        Send raw data and magnitudes, this runs the channel encoder as well as
        the thresholder.
        """
        sdr = self.channel.encode(img_data)
        assert(magnitude.shape == self.channel.input_shape)
        # Reshape to broadcast magnitudes across the data dimension to all
        # samples and their thresholds.
        magnitude = magnitude.reshape(magnitude.shape + (1,))
        sdr[magnitude < self.thresholds] = False
        return sdr


class EyeSensorParameters(Parameters):
    parameters = [
        'eye_dimensions',
        'fovea_param_1',
        'fovea_param_2',
        'min_scale',
        'max_scale',
        'num_samples',
        'hue_sparsity',
        'sat_sparsity',
        'val_sparsity',
        'edge_encoder',
    ]
    def __init__(self,
        eye_dimensions  = (512, 512),
        fovea_param_1   = .05,
        fovea_param_2   = 20,
        min_scale       =  1,
        max_scale       = 10,
        num_samples     = 20,
        hue_sparsity    = .50,
        sat_sparsity    = .50,
        val_sparsity    = .50,
        edge_encoder    = None,):
        """
        Argument  eye_dimensions ...
        Arguments fovea_param_1 and fovea_param_2 ...
        Arguments min_scale and max_scale ...
        Arguments num_samples, hue_sparsity, sat_sparsity and val_sparsity are
                  for channel encoders.
        Argument edge_encoder is an instance of ChannelThresholderParameters.
        """
        # Don't create objects in the default arguments because default
        # arguments are created exactly once at startup and reused every other
        # time.  This resuse directly causes critical data loss in the genetics
        # module.
        if edge_encoder is None:
            edge_encoder = ChannelThresholderParameters()
        # Get the parent class to save all these parameters.
        super().__init__(**{k:v for k,v in locals().items() if k != 'self'})

# EXPERIMENT: Add a motion sensor to the EyeSensor.
#
# TODO: Should preprocess_edges find the edges before or after casting to
# greyscale?  Currently this finds edges after casting to greyscale so it really
# is only finding the edges in the value channel.  If instead I found the edges
# in all RGB channels and then averaged their magnitudes if could detect edges
# in Hue, Saturation and Value channels.  How would it handle the edge angles?
# Maybe just take the angle from the RGB channel with the greatest magnitude?
class EyeSensor:
    """
    Optic sensor with central fovae.

    This sensor has the following design criteria:
    1) The central fovae should be capable of identifying objects in 1 or 2 views.
    2) The periferal vision should be capable of locating and tracking objects.

    This eye has 4 degrees of freedom: X and Y location, scale, and orientation.
    These values can be assigned to directly or by the EyeController class.

    Fun Fact 1: The human optic nerve has 800,000 ~ 1,700,000 nerve fibers.
    Fun Fact 2: The human eye can distiguish between 10 million different colors.
    Sources: Wikipedia.
    """
    def __init__(self, parameters):
        """
        Attribute optic_sdr ... retina's output
        Attribute rgb ... The most recent view, kept as a attribute.

        Attribute position     (X, Y) coords of eye within image, Read/Writable
        Attribute orientation  ... units are radians, Read/Writable
        Attribute scale        ... Read/Writable

        Private Attributes:
            self.eye_coords.shape = (2, view-x, view-y)
            self.eye_coords[input-dim, output-coordinate] = input-coordinate
        """
        self.args = args    = parameters
        self.eye_dimensions = tuple(int(round(ed)) for ed in args.eye_dimensions)
        self.eye_coords     = EyeSensor.complex_eye_coords(self.eye_dimensions,
                                        args.fovea_param_1, args.fovea_param_2)
        self.hue_encoder = ChannelEncoder(  input_shape = self.eye_dimensions,
                                            num_samples = args.num_samples,
                                            sparsity    = args.hue_sparsity,
                                            dtype       = np.float32,
                                            drange      = range(0,360),
                                            wrap        = True,)
        self.sat_encoder = ChannelEncoder(  input_shape = self.eye_dimensions,
                                            num_samples = args.num_samples,
                                            sparsity    = args.sat_sparsity,
                                            dtype       = np.float32,
                                            drange      = (0, 1),
                                            wrap        = False,)
        self.val_encoder = ChannelEncoder(  input_shape = self.eye_dimensions,
                                            num_samples = args.num_samples,
                                            sparsity    = args.val_sparsity,
                                            dtype       = np.float32,
                                            drange      = (0, 1),
                                            wrap        = False,)
        self.edge_encoder = ChannelThresholder(args.edge_encoder,
                                            input_shape = self.eye_dimensions,
                                            dtype       = np.float32,
                                            drange      = (-math.pi, math.pi),
                                            wrap        = True)

        depth = self.hue_encoder.output_shape[-1] + self.edge_encoder.output_shape[-1]
        self.optic_sdr = SDR(self.eye_dimensions + (depth,))

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

    def center_view(self):
        """Center the view over the image"""
        self.orientation = 0
        self.position    = np.divide(self.image.shape[:2], 2)
        self.scale       = np.max(np.divide(self.image.shape[:2], self.eye_dimensions))

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
        color_sdr   = np.logical_and(np.logical_and(hue_sdr, sat_sdr), val_sdr)

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

        self.optic_sdr.dense = np.dstack([color_sdr, edge_sdr])
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

    def view_samples(self, show=True):
        """Displays the samples."""
        if not self.samples:
            return  # Nothing to show...
        plt.figure("Sample views")
        num = len(self.samples)
        rows = math.floor(num ** .5)
        cols = math.ceil(num / rows)
        for idx, img in enumerate(self.samples):
            plt.subplot(rows, cols, idx+1)
            plt.imshow(img, interpolation='nearest')
        if show:
            plt.show()


# EXPERIMENT: Try breaking out each output encoder by type instead of
# concatenating them all together.  Each type of sensors would then get its own
# HTM.  Maybe keep the derivatives with their source?
#
class EyeControllerParameters(Parameters):
    parameters = [
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
        num_cv                      = 600,
        pos_stddev                  = 1,
        angle_stddev                = math.pi / 8,
        scale_stddev                = 2,
        position_encoder            = None,
        velocity_encoder            = None,
        angle_encoder               = None,
        angular_velocity_encoder    = None,
        scale_encoder               = None,
        scale_velocity_encoder      = None,):
        """
        Argument num_cv is the approximate number of control vectors to use.
        Arguments pos_stddev, angle_stddev, and scale_stddev are the standard
                  deviations of the control vector movements, control vectors
                  are normally distributed about a mean of 0.

        Arguments position_encoder, velocity_encoder, angle_encoder,
                  angular_velocity_encoder, scale_encoder, and
                  scale_velocity_encoder are instances of
                  RandomDistributedScalarEncoderParameters.
        """
        # Motor Sensor Defaults.  Don't create objects in the default parameters
        # because default arguments are created exactly once at startup and
        # reused every other time.  This resuse directly causes critical data
        # loss in the genetics module.
        if position_encoder is None:
            position_encoder        = RandomDistributedScalarEncoderParameters()
        if velocity_encoder is None:
            velocity_encoder        = RandomDistributedScalarEncoderParameters()
        if angle_encoder is None:
            angle_encoder           = RandomDistributedScalarEncoderParameters(
                                            resolution  = math.pi / 80,)
        if angular_velocity_encoder is None:
            angular_velocity_encoder = RandomDistributedScalarEncoderParameters(
                                            resolution  = math.pi / 80,)
        if scale_encoder is None:
            scale_encoder           = RandomDistributedScalarEncoderParameters(
                                            resolution  = .2,)
        if scale_velocity_encoder is None:
            scale_velocity_encoder  = RandomDistributedScalarEncoderParameters(
                                            resolution  = .2,)
        # Get the parent class to save all these parameters as attributes under
        # the same names.
        super().__init__(**{k:v for k,v in locals().items() if k != 'self'})

class EyeController:
    """
    Motor controller for the EyeSensor class.

    The eye sensor has 4 degrees of freedom: X and Y location, scale, and
    orientation. These values can be controlled by activating control vectors,
    each of which  has a small but cumulative effect.  CV's are normally
    distributed with a mean of zero.  Activate control vectors by calling
    controller.move(control-vectors).

    The controller outputs its current location, scale and orientation as well
    as their first derivatives w/r/t time as an SDR.
    """
    def __init__(self, parameters, eye_sensor):
        """
        Attribute control_sdr ... eye movement input controls
        Attribute motor_sdr ... internal motor sensor output

        Attribute gaze is a list of tuples of (X, Y, Orientation, Scale)
                  History of recent movements, self.move() updates this.
                  This is cleared by the following methods:
                      self.new_image() 
                      self.center_view()
                      self.randomize_view()
        """
        assert(isinstance(parameters, EyeControllerParameters))
        assert(isinstance(eye_sensor, EyeSensor))
        self.args = args = parameters
        self.eye_sensor  = eye_sensor
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
        self.gaze = []

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
        eye = self.eye_sensor
        # Calculate the forces on the motor
        controls = self.control_vectors[self.control_sdr.index]
        controls = np.sum(controls, axis=0)
        dx, dy, dangle, dscale = controls
        # Calculate the new rotation
        eye.orientation = (eye.orientation + dangle) % (2*math.pi)
        # Calculate the new scale
        new_scale  = np.clip(eye.scale + dscale, eye.args.min_scale, eye.args.max_scale)
        real_ds    = new_scale - eye.scale
        avg_scale  = (new_scale + eye.scale) / 2
        eye.scale = new_scale
        # Scale the movement such that the same CV yields the same visual
        # displacement, regardless of scale.
        dx       *= avg_scale
        dy       *= avg_scale
        # Calculate the new position.  
        x, y     = eye.position
        p        = [x + dx, y + dy]
        p        = np.clip(p, [0,0], eye.image.shape[:2])
        real_dp  = np.subtract(p, eye.position)
        eye.position = p
        # Book keeping.
        self.gaze.append(tuple(eye.position) + (eye.orientation, eye.scale))
        # Put together information about the motor.
        velocity = (
            eye.position[0],
            eye.position[1],
            eye.orientation,
            eye.scale,
            real_dp[0],
            real_dp[1],
            dangle,
            real_ds,
        )
        # Encode the motors sensors and concatenate them into one big SDR.
        v_enc = [enc.encode(v) for v, enc in zip(velocity, self.motor_encoders)]
        self.motor_sdr.dense = np.concatenate([sdr.dense for sdr in v_enc])
        return self.motor_sdr

    def reset_gaze_tracking(self):
        """
        Discard any prior gaze tracking.  Call this after forcibly moving eye
        to a new starting position.
        """
        self.gaze = [(
            self.eye_sensor.position[0],
            self.eye_sensor.position[1],
            self.eye_sensor.orientation,
            self.eye_sensor.scale)]
        # Update motor sensors, set the motor velocity to zero.
        self.control_sdr.zero()
        self.move()

    def gaze_tracking(self, diag=True):
        """
        Returns vector of tuples of (position-x, position-y, orientation, scale)
        """
        if diag:
            im   = PIL.Image.fromarray(self.eye_sensor.image)
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
                    corners.append(self.eye_sensor.eye_coords[:, ec_x, ec_y])
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
