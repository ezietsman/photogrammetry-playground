import docopt
from detectors.DefaultTargetDetector import DefaultDetector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

__doc__ = '''\
Usage: test-DefaultTargetDetector.py FILE [--plot]
'''
if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    detector = DefaultDetector(arguments['FILE'])
    detector.find_targets()

    for rad in detector.radtargets:
        print rad.encoding

    print("RAD targets: {N}".format(N=len(detector.radtargets)))
    print("Ring targets: {N}".format(N=len(detector.smalltargets)))

    if arguments['--plot']:
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, aspect='equal')
        plt.imshow(detector.threshold, cmap=cm.gray, interpolation='nearest')

        for small in detector.smalltargets:
            (x, y), (Ma, ma), angle = small.ellipse
            ell = Ellipse([x, y], Ma, ma, angle,
                          facecolor='none',
                          edgecolor='g',
                          linewidth=2)
            ax1.add_artist(ell)

        for rad in detector.radtargets:
            (x, y), (Ma, ma), angle = rad.ellipse
            ell = Ellipse([x, y], Ma, ma, angle,
                          facecolor='none',
                          edgecolor='r')
            ax1.add_artist(ell)
        plt.show()
