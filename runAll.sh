#!/bin/bash
algorithm="dropout"
for idtest in 0 1 2 3 4
do
	for seed in {1..10}
	do
		if [[ $algorithm == "deterministicGaussian" ]]
		then
			python tests/train_deterministic_gaussian.py --id-test=$idtest --seed=$seed
			python tests/test_calibration.py --id-test=$idtest --seed=$see --test_name="deterministicGaussian"
		elif [[ $algorithm == "dropout" ]]
		then
			python scripts/train_dropout.py --pickle --id-test=$idtest --seed=$seed
			python tests/test_calibration.py --id-test=$idtest --seed=$seed --test_name="dropout"
		fi
	done
done
		#python tests/train_ensembles --pickle --id-test=2
		#python tests/train_bitrap_BT --pickle --id-test=2
		#python tests/test_calibration --gaussian-output --calibration-conformal --test-name:"deterministicGaussian"
		#python tests/test_calibration --calibration-conformal --test-name:"ensembles"
		#python tests/train_ensembles --pickle --id-test=2

		#python tests/test_calibration --gaussian-output --calibration-conformal --test-name:"ensembles"
		#python tests/test_calibration --calibration-conformal --test-name:"dropout"

		#python tests/test_calibration --gaussian-output --calibration-conformal --test-name:"dropout"
		#python tests/test_calibration --calibration-conformal --test-name:"bitrap"

		#python tests/train_bitrap_BT --pickle --id-test=2
		#python tests/test_calibration --gaussian-output --calibration-conformal --test-name:"bitrap"

		#python tests/train_deterministic --pickle --id-test=2
		#python tests/train_dropout --pickle --id-test=2
