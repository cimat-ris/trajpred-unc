from errno import EEXIST
from os import makedirs, path
from trajpred_unc.utils.constants import IMAGES_DIR

def mkdir_p(mypath):
    """
    Creates a directory and, if needed, all parent directories.
    """
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

class Output_directories:
    def __init__(self):
        self.trajectories = path.join(IMAGES_DIR, "trajectories")
        self.calibration = path.join(IMAGES_DIR, "calibration")
        self.metrics = path.join(self.calibration, "metrics")
        self.confidence = path.join(self.calibration, "confidence_level")
        self.hdr = path.join(IMAGES_DIR, "HDR1")
        self.hdr2 = path.join(IMAGES_DIR, "HDR2")
        self.trajectories_kde = path.join(IMAGES_DIR, "trajectories_kde")
        mkdir_p(self.trajectories)
        mkdir_p(self.calibration)
        mkdir_p(self.metrics)
        mkdir_p(self.confidence)
        mkdir_p(self.hdr)
        mkdir_p(self.hdr2)
        mkdir_p(self.trajectories_kde)
