 #!/bin/bash
for idTest in 'bitrap_np_hotel.yml' 'bitrap_np_eth.yml' 'bitrap_np_zara1.yml' 'bitrap_np_zara2.yml' 'bitrap_np_univ.yml'
do
		echo $idTest
		for ensembleId in 1 2 3 4 5
		do
			echo $ensembleId
			python tests/train_bitrap.py --seed $ensembleId --config_file $idTest
		done
done
