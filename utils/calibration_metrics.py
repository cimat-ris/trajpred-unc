import numpy as np
import logging, os
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
import time
from utils.plot_utils import plot_calibration_curves

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
	mace                 = np.mean(abs_diff_proportions)
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
	rmsce                    = np.sqrt(np.mean(squared_diff_proportions))
	return rmsce

def compute_calibration_metrics(exp_proportions, obs_proportions, metrics_data, position, key):
	"""
	Compute MA, MACE and RMSCE calibration metrics and save those into metrics_data dictionary
	Args:

	Returns:
	"""
	ma    = miscalibration_area(exp_proportions, obs_proportions)
	mace  = mean_absolute_calibration_error(exp_proportions, obs_proportions)
	rmsce = root_mean_squared_calibration_error(exp_proportions, obs_proportions)
	metrics_data.append([key + " pos " + str(position),mace,rmsce,ma])
	logging.info("{}:  MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(key,mace,rmsce,ma))

def generate_metrics_curves(conf_levels, unc_pcts, cal_pcts, metrics, position, method, output_dirs, prediction_method_name, suffix="cal"):
	"""
	Produce calibration curves and compute calibration metrics.
	Args:

	Returns:
	"""
	# Evaluate metrics before/after calibration
	compute_calibration_metrics(conf_levels, unc_pcts, metrics, position, "Before Recalibration")
	compute_calibration_metrics(conf_levels, cal_pcts, metrics, position, "After  Recalibration")
	# Save plot_calibration_curves
	output_image_name = os.path.join(output_dirs.confidence, "confidence_level_"+suffix+"_"+prediction_method_name+"_method_"+str(method)+"_pos_"+str(position)+"_"+ str(time.time())+".pdf")
	print(output_image_name)
	plot_calibration_curves(conf_levels, unc_pcts, cal_pcts, output_image_name)
