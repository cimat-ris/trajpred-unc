import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

def miscalibration_area(
	exp_proportions: np.ndarray,
	obs_proportions: np.ndarray
) -> float:
	"""Miscalibration area.
	This is identical to mean absolute calibration error and ECE, however
	the integration here is taken by tracing the area between curves.
	In the limit of num_bins, miscalibration area and
	mean absolute calibration error will converge to the same value.
	Args:

	Returns:
		A single scalar which calculates the miscalibration area.
	"""

	# Compute approximation to area between curves
	polygon_points = []
	for point in zip(exp_proportions, obs_proportions):
		polygon_points.append(point)
	for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
		polygon_points.append(point)
	polygon_points.append((exp_proportions[0], obs_proportions[0]))
	polygon = Polygon(polygon_points)
	x, y = polygon.exterior.xy
	ls = LineString(np.c_[x, y])
	lr = LineString(ls.coords[:] + ls.coords[0:1])
	mls = unary_union(lr)
	polygon_area_list = [poly.area for poly in polygonize(mls)]
	miscalibration_area = np.asarray(polygon_area_list).sum()

	return miscalibration_area

def mean_absolute_calibration_error(
	 exp_proportions: np.ndarray,
	 obs_proportions: np.ndarray
) -> float:
	"""Mean absolute calibration error; identical to ECE.
	Args:

	Returns:
		A single scalar which calculates the mean absolute calibration error.
	"""

	abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
	mace = np.mean(abs_diff_proportions)

	return mace

def root_mean_squared_calibration_error(
	 exp_proportions: np.ndarray,
	 obs_proportions: np.ndarray
) -> float:
	"""Root mean squared calibration error.
	Args:

	Returns:
		A single scalar which calculates the root mean squared calibration error.
	"""

	squared_diff_proportions = np.square(exp_proportions - obs_proportions)
	rmsce = np.sqrt(np.mean(squared_diff_proportions))

	return rmsce
