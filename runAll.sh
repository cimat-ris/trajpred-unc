#!/bin/bash
algorithm="bitrap"
for idtest in 0 1 2 3 4
do
	for seed in {1..10}
	do
		if [[ $algorithm == "deterministicGaussian" ]]
		then
			python scripts/train_deterministic_gaussian.py --id-test=$idtest --seed=$seed
			python scripts/test_calibration.py --id-test=$idtest --seed=$see --test-name="deterministicGaussian" --gaussian_output
		elif [[ $algorithm == "dropout" ]]
		then
			python scripts/train_dropout.py --id-test=$idtest --seed=$seed
			python scripts/test_calibration.py --id-test=$idtest --seed=$seed --test-name="dropout"  --gaussian_output
		elif [[ $algorithm == "variational" ]]
		then
			python scripts/train_variational.py --id-test=$idtest --seed=$seed
			python scripts/test_calibration.py --id-test=$idtest --seed=$seed --test-name="variational"  --gaussian_output
		elif [[ $algorithm == "bitrap" ]]
		then
			python scripts/train_bitrap.py --id-test=$idtest --seed=$seed
			python scripts/test_bitrap.py --id-test=$idtest --seed=$seed
			python scripts/test_calibration.py --id-test=$idtest --seed=$seed --test-name="bitrap" --absolute-coords
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
