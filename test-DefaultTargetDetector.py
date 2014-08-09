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

#   print("RAD targets: {N}".format(N=len(detector.radtargets)))
#   print("Ring targets: {N}".format(N=len(detector.smalltargets)))

#   if arguments['--plot']:
#       fig = plt.figure(figsize=(12, 12))
#       ax1 = fig.add_subplot(111, aspect='equal')
#       plt.imshow(detector.threshold, cmap=cm.gray, interpolation='nearest')

#       for sq in detector.smallpolys:
#           vertices = sq[:]
#           vertices = np.append(vertices, vertices[0]).reshape((5,2))
#           plt.plot(vertices[:,0], vertices[:,1], 'r-', linewidth=2)

#       for ell in detector.ellipses:
#           (x, y), (Ma, ma), angle = ell
#           ell = Ellipse([x, y], Ma, ma, angle,
#                         facecolor='none',
#                         edgecolor='b',
#                         linewidth=2)
#           ax1.add_artist(ell)


#       for ell in detector.radtargets:
#           (x, y), (Ma, ma), angle = ell
#           ell = Ellipse([x, y], Ma, ma, angle,
#                         facecolor='none',
#                         edgecolor='r')
#           ax1.add_artist(ell)
#       plt.show()
