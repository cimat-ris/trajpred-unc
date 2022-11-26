# trajpred-bdl

## Training

To train the deterministic model:

```
python tests/train_torch_deterministic.py
```

To train the deterministic model with variances as output:

```
 python3 tests/train_torch_deterministic_with_variances.py
```

To train the ensemble model and calibrate the uncertainties:

```
python tests/train_torch_ensembles_calibration.py
```

## Testing

```
python3 tests/train_torch_deterministic_with_variances.py  --no-retrain --pickle  --examples 10
```

## Calibration: a postprocess step

After a model is trained, it saves it's results in a `pickle` file, then the calibration for it uses the output from a trained model and can be executed as follows:

```
# training the desired model
$ python tests/train_torch_deterministic_gaussian.py --pickle --no-retrain

# calibration postprocess
$ python tests/test_calibration.py --test-name="deterministicGaussian" --gaussian-isotonic
... alphas computation prints ...
Before Recalibration:  MACE: 0.21261, RMSCE: 0.25398, MA: 0.22324
After Recalibration:   MACE: 0.00417, RMSCE: 0.00511, MA: 0.00381
Before Recalibration:  MACE: 0.30880, RMSCE: 0.35925, MA: 0.32420
After Recalibration:   MACE: 0.06962, RMSCE: 0.08891, MA: 0.07237
                      0             1
0            calibrated  uncalibrated
1  [-2.402059933196499]     -2.320702
```
Where `--test-name` is a key to retrieved the previos pickle file saved, available keys:
* `deterministicGaussian`: for `train_torch_deterministic_gaussian.py` model.
* `ensembles`: for `train_torch_ensembles_calibration.py` model.
* `dropout`: for `train_torch_dropout_calibration.py` model.
* `bitrap`: for `train_torch_bitrap_BT.py` model.

The `test_calibration.py` script uses Isotonic regression to compute the calibration metrics by default, it can use conformal calibration too by passing the arg `--calibration-conformal`. Also, it can be specified a *gaussian* argument for each one: `--gaussian-isotonic` and `--gaussian-conformal`, respectively.

## To run/evaluate Bitrap

* Clone the [bitrap repository](https://github.com/umautobots/bidireaction-trajectory-prediction).
* The train/test partition from the [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) repository are now present in the datasets/trj++ directory as .pkl files.
* Modify *bitrap_np_ETH.yml* lines 30 and set the path to where the .json file is located. You may also change BATCH_SIZE or NUM_WORKERS   
* To train bitrap, run
```
python tests/train_bitrap.py --config_file bitrap_np_ETH.yml --seed n
```
By changing the seed, you will be building different models for an ensemble.
* To generate data calibration from bitrap, run
```
python tests/test_bitrap.py
```
