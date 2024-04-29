#!/bin/bash
algorithm="deterministicGaussian"
for idtest in 0 1 2 3 4
do
	for seed in {1..10}
	do
		if [[ $algorithm == "deterministicGaussian" ]]
		then
			python scripts/train_deterministic_gaussian.py --id-test=$idtest --seed=$seed
			python scripts/test_calibration.py --id-test=$idtest --seed=$see --test-name="deterministicGaussian" --gaussian-output
		elif [[ $algorithm == "dropout" ]]
		then
			python scripts/train_dropout.py --id-test=$idtest --seed=$seed
			python scripts/test_calibration.py --id-test=$idtest --seed=$seed --test-name="dropout"  --gaussian-output
		elif [[ $algorithm == "variational" ]]
		then
			python scripts/train_variational.py --id-test=$idtest --seed=$seed
			python scripts/test_calibration.py --id-test=$idtest --seed=$seed --test-name="variational"  --gaussian-output
		elif [[ $algorithm == "bitrap" ]]
		then
			python scripts/train_bitrap.py --id-test=$idtest --seed=$seed
			python scripts/test_bitrap.py --id-test=$idtest --seed=$seed
			python scripts/test_calibration.py --id-test=$idtest --seed=$seed --test-name="bitrap" --absolute-coords
		elif [[ $algorithm == "agentformer" ]]
		then
			python scripts/test_agentformer.py --id-test=$idtest --seed=$seed
			python scripts/test_calibration.py --id-test=$idtest --seed=$seed --test-name="agentformer" --absolute-coords
		fi
	done
done
