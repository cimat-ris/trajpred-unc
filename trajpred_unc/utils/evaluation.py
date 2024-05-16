import logging
import numpy as np
import pandas as pd
import os

# Perform quantitative evaluation
def evaluation_minadefde(predictions_samples,targets, model_name):
	logging.info("----> Predictions: {}".format(predictions_samples.shape))
	logging.info("----> Ground truth: {}".format(targets.shape))
	# All squared differences
	diff = targets[:,:,:2].detach().unsqueeze(1).numpy() - (predictions_samples)
	diff = diff**2
	# Euclidean distances
	diff = np.sqrt(np.sum(diff, axis=3))
	# minADEs for each data point 
	ade  = np.min(np.mean(diff,axis=2), axis=1)
	# minFDEs for each data point 
	fde  = np.min(diff[:,:,-1], axis=1)
	results = [["mADE", "mFDE"], [np.mean(ade), np.mean(fde)]]    
	# Save results into a csv file
	output_csv_name = "images/calibration/" + model_name +"_min_ade_fde.csv"
	df = pd.DataFrame(results)
	df.to_csv(output_csv_name, mode='a', header=not os.path.exists(output_csv_name))
	print(df)
	return results	
