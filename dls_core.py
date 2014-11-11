# Copyright 2013, Jerome Fung, Nicholas Schade, Vinothan N. Manoharan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Core objects for working with correlation functions and metadata

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""
import numpy as np
import matplotlib.pyplot as plt

class CorrFn(object):
    """
    A generic class to hold correlation function data.  
    
    Parameters
    ----------
    delay_times : array_like
        Delay times in seconds
    data : array_like
        Values of correlation function at delay times.  This should be
        equal to C = (g2-B), where B is the baseline
    baseline : float, optional
        Value of the baseline B that was subtracted from g2.  Default:
        1.0 (normalized to calculated baseline)
    duration : float, optional
        Duration of measurement, in seconds.  Can be used to infer
        statistical significance of measured correlation function.
    samples : integer, optional
        Number of runs that were averaged together to produce the
        resulting correlation function.  Can be used to infer
        statistical significance of measured correlation function.
    count_rate : float, optional
        Average detector count rate during the measurement.  Can be
        used to determine statistical significance of measurement.
    """
    def __init__(self, delay_times, data, baseline=1.0, duration=None,
                 samples=None, count_rate=None):
        self.delay_times = delay_times
        self.data = data
        self.baseline = baseline
        self.duration = duration
        self.samples = samples
        self.count_rate = count_rate

    def plot(self, offset=1.0, **kwargs):
        """
        Makes a nice plot of the correlation function

        Parameters
        ----------
        offset : float
            offset on log-log plot for correlation function
        **kwargs : list
            keyword arguments to pass to plot()
        """
        plt.loglog(tau, offset*self.data)
        plt.xlabel('tau (s)')
        if offset != 1.0:
            plt.ylabel(u'g^2(tau)-1, with vertical offsets')
        else:
            plt.ylabel(u'g^2(tau)-1')

    def g2(self):
        """
        Returns unnormalized version of correlation function
        """
        # TODO: fix this after figuring out how to get baseline from
        # instruments.  ST100 seems to report g2, so this should be
        # safe for ST100.  Malvern though seems to report C=g2-B.  I
        # assume they are using the calculated baseline (1.0) for B,
        # but I'm not certain.

        return CorrFn(self.delay_times, (self.data+self.baseline))

    def trimmed(self):
        """
        Returns trimmed version of correlation function, without the
        zeros that correspond to longer time lags than data was
        collected for.

        Notes
        -----
        This will remove all identically zero values in the
        correlation function, and eliminate the corresponding delay
        times. 
        """
        # depending on baseline subtraction, zeros could show up as
        # 0.0 or -1.0 in the stored correlation function
        zeros = np.where(self.data == 0.0)[0]
        negs = np.where(self.data == -1.0)[0]
        times = np.delete(self.delay_times, zeros)
        times = np.delete(self.delay_times, negs)
        data = np.delete(self.data, zeros)
        data = np.delete(self.data, negs)
        return CorrFn(times, data, duration=self.duration,
                      samples=self.samples, count_rate=self.count_rate)

    def trimmed_t(self, tmin = 0., tmax = 1e6):
        '''
        Returns trimmed version of correlation, eliminating delay times
        less than tmin and greater than tmax.
        '''
        condition = (self.delay_times > tmin) * (self.delay_times < tmax)
        data_trimmed = self.data[condition]
        times_trimmed = self.delay_times[condition]
        return CorrFn(times_trimmed, data_trimmed, duration = self.duration,
                      samples = self.samples, count_rate = self.count_rate)


PERPENDICULAR_POLARIZATION = (1.0, 0.0)
PARALLEL_POLARIZATION = (0.0, 1.0)

class Optics(object):
    """
    Optical parameters used to make a measurement.  In many cases
    these will be constant for a given instrument, so that you can
    save and reuse the metadata for different measurements.

    Parameters
    ----------
    wavelen : float, optional
        Wavelength of laser (in vacuum)
    index : float, optional
        Refractive index of medium
    heterodyne : boolean, optional
        Set to True if instrument uses heterodyne detection.  Default: False
        (homodyne detection)
    tilt : float, optional
        Angle between the scattering plane and the detector, in
        degrees.  Default is 0 degrees.  Some instruments put the
        detector below the scattering plane, which can affect q-vector
        calculations and polarization
    incident_pol : 2-tuple (float, float), optional
        Perpendicular and parallel components of polarization.
        Default is to assume (1.0, 0.0), or light polarized
        perpendicularly to the scattering plane
    detector_pol : 2-tuple (float, float), optional
        Perpendicular and parallel components of polarization measured
        at the detector. Default is to assume (1.0, 0.0), or light polarized
        perpendicularly to the scattering plane
    """
    def __init__(self, wavelen=None, index=None, heterodyne=False, 
                 tilt = 0.0,
                 incident_pol=PERPENDICULAR_POLARIZATION, 
                 detector_pol=PERPENDICULAR_POLARIZATION):
        self.wavelen = wavelen
        self.index = index
        self.heterodyne = heterodyne
        self.tilt = tilt
        self.incident_pol = incident_pol
        self.detector_pol = detector_pol

    def polarization(self, theta):
        """
        Calculates incident polarization as a function of scattering
        angle.  This will deviate significantly from a constant only
        when the detector angle is close to or less than tilt angle.

        Parameters
        ----------
        theta : float
            in-plane scattering angle (theta)

        Returns
        -------
        (float, float)
            2-tuple of (parallel, perpendicular) components of
            polarization 
        """
        # TODO: add a function here to calculate polarization as a function of
        # scattering angle and tilt angle
        return self.incident_pol

    def angle(self, theta):
        """
        Returns actual scattering angle as a function of in-plane
        scattering angle theta.  

        Parameters
        ----------
        theta : float
            in-plane scattering angle (theta)
        """
        # TODO: check to see if SciTech returns actual scattering
        # angle, not just theta.  If so, this function is probably
        # unnecessary
        return theta
    
    def qsca(self, theta):
        """
        Returns magnitude of scattering wavevector (or
        momentum-transfer vector) q.  

        Parameters
        ----------
        theta : float
            in-plane scattering angle (theta)
        """
        angle = self.angle(theta)*np.pi/180. 
        return 4*np.pi*self.index*np.sin(angle/2.)/self.wavelen

class Measurement(object):
    """
    A Measurement is a correlation function with associated metadata,
    including the scattering angle, temperature, sample parameters,
    and optical parameters of the instrument.

    Parameters
    ----------
    corrfn: `dls.CorrFn` object
        Measured correlation function
    angle : float
        Scattering angle, in degrees
    optics : Optics object 
        Optical parameters used in measurement
    temperature : float, optional
        Absolute temperature
    metadata : dictionary, optional
        Dictionary of other metadata about the measurement.  This is
        generally detected by the function that reads in the data.  It
        might include keys such as "id" (name of measurement or
        sample) or "timestamp" (date and time of measurement)

    Attributes
    ----------
    qsca : float
        Value of scattering wavevector (in units of inverse length)

    """
    def __init__(self, corrfn, angle, optics, temperature=None, 
                 metadata=None): 
        self.corrfn = corrfn
        self.angle = angle
        self.optics = optics
        self.temperature = temperature
        self.metadata = metadata

    def plot(self, offset=1.0, **kwargs):
        """
        Makes a nice plot of the correlation function, 

        Parameters
        ----------
        offset : float
            offset on log-log plot for correlation function
        **kwargs : list
            keyword arguments to pass to plot()
        """
        self.corrfn.plot() 
        
    @property
    def qsca(self):
        """
        Returns value of scattering wavevector (in units of inverse
        length)
        """
        return self.optics.qsca(self.angle)


    
