"""
constants.py:  Constants and data structures used in this package
"""
import os


class PhyConst(object):
    """
    Physical constants
    """
    light_speed = 2.99792458e8
    kT = -174


class DataConfig(object):
    """
    Meta data from MATLAB simulation
    """

    def __init__(self):
        self.fc = 140e9
        self.date_created = 'unknown'
        self.desc = 'data set'
        self.rx_types = ['RX0']

    def __str__(self):
        string = ('Description:   %s' % self.desc) + os.linesep
        string += ('Date created:  %s' % self.date_created) + os.linesep
        string += ('fc:            %12.4e Hz' % self.fc) + os.linesep
        return string

    def summary(self):
        """
        Prints a summary of the configuration
        """
        print(str(self))