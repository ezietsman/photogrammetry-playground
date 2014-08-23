import docopt
import TargetDetector

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.rcParams['font.family'] = 'PT Sans'
# need matplotlib >= 1.4
matplotlib.style.use('ggplot')


__doc__ = '''\
Usage: test-DefaultTargetDetector.py FILE [--plot] [--save]
'''
if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    detector = TargetDetector.DefaultDetector(arguments['FILE'])
    detector.find_targets()

    print("RAD targets: {N}".format(N=len(detector.radtargets)))
    print("Ring targets: {N}".format(N=len(detector.smalltargets)))

    if arguments['--plot']:
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, aspect='equal')
        plt.imshow(detector.image, cmap=cm.gray, interpolation='nearest')

#       for sq in detector.square_contours:
#           x = sq.vertices[:,:,0]
#           y = sq.vertices[:,:,1]
#           plt.plot(x, y, 'y', lw=2)

#       for ell in detector.ellipses:
#           (x, y), (Ma, ma), angle = ell
#           ell = Ellipse([x, y], Ma, ma, angle,
#                         facecolor='none',
#                         edgecolor='g',
#                         linewidth=2)
#           ax1.add_artist(ell)

        for ell in detector.smalltargets:
            (x, y), (Ma, ma), angle = ell
            ell = Ellipse([x, y], Ma, ma, angle,
                          facecolor='none',
                          edgecolor='b',
                          linewidth=2)
            ax1.add_artist(ell)

        for ell in detector.radtargets:
            (x, y), (Ma, ma), angle = ell
            ell1 = Ellipse([x, y], Ma, ma, angle, facecolor='none',
                           edgecolor='r', lw=3)
            ax1.add_artist(ell1)
            # smaller ellipse for blog
#           ell2 = Ellipse([x, y], 0.85*Ma, 0.85*ma, angle, facecolor='none',
#                          edgecolor='y', lw=3)
#           ax1.add_artist(ell2)

#           ell3 = Ellipse([x, y], 0.6*Ma, 0.6*ma, angle, facecolor='none',
#                          edgecolor='g', lw=3)
#           ax1.add_artist(ell3)

        if arguments['--save']:
            plt.savefig('test.png', bbox_inches=('tight'))
        plt.show()


