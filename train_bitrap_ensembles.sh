 #!/bin/bash
for idTest in 'bitrap_np_hotel.yml' 'bitrap_np_eth.yml' 'bitrap_np_zara1.yml' 'bitrap_np_zara2.yml' 'bitrap_np_univ.yml'
do
		echo "Dataset: $idTest"
		for ensembleId in 1 2 3 4 5
		do
			echo "Ensemble id: $ensembleId"
			python tests/train_bitrap.py --seed $ensembleId --config_file configs/$idTest
		done
done
