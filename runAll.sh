#!/bin/bash
for idtest in 0 1 2 3 4
do
		python tests/train_deterministic_gaussian.py --id-test=$idtest
		python tests/test_calibration.py --id-test=$idtest
done

		#python tests/train_ensembles --pickle --id-test=2
		#python tests/train_bitrap_BT --pickle --id-test=2
		#python tests/test_calibration --gaussian-output --calibration-conformal --test-name:"deterministicGaussian"
		#python tests/test_calibration --calibration-conformal --test-name:"ensembles"
		#python tests/train_ensembles --pickle --id-test=2

		#python tests/test_calibration --gaussian-output --calibration-conformal --test-name:"ensembles"
		#python tests/test_calibration --calibration-conformal --test-name:"dropout"
		#python tests/train_dropout --pickle --id-test=2

		#python tests/test_calibration --gaussian-output --calibration-conformal --test-name:"dropout"
		#python tests/test_calibration --calibration-conformal --test-name:"bitrap"

		#python tests/train_bitrap_BT --pickle --id-test=2
		#python tests/test_calibration --gaussian-output --calibration-conformal --test-name:"bitrap"

		#python tests/train_deterministic --pickle --id-test=2
		#python tests/train_dropout --pickle --id-test=2
