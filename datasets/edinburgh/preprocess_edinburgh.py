import os, sys
import pandas as pd
import matplotlib.pyplot as plt

from opentraj.loaders.loader_edinburgh import load_edinburgh
sys.path.append('.')

months = {
    #'Jan' : {'01','02','04','05','06','07','08','10','11','12','13','14','15','16','17','18','19'},
    #'May' : {'29','30','31'},
    #'Jun' : {'02','03','04','05','06','08','09','11','12','14','16','17','18','20','22','24','25','26','29','30'},
    #'Jul' : {'01','02','04','11','12','13','14','17','18','19','20','21','22','23','25','26','27','28','29','30'},
    #'Aug' : {'01','24','25','26','27','28','29','30'},
    'Sep' : {'01','02','04','05','06','10','11','12','13','14','16','18','19','20','21','22','23','24','25','27','28','29','30'},
    'Oct': {'02','03','04','05','06','07','08','09','10','11','12','13','14','15'},
    'Dec' : {'06','11','14','15','16','18','19','20','21','22','23','24','29','30','31'}
}
# Fixme: set proper OpenTraj directory
edi_root = os.path.join('../../../OpenTraj', 'datasets', 'Edinburgh','annotations')
edi_data = {key: pd.DataFrame() for key in months.keys() }

for month, videos_per_month in months.items():
    scene_video_ids         = [day+month for day in videos_per_month]
    traj_datasets_per_scene = []

    for scene_video_id in scene_video_ids:
        annot_file = os.path.join(edi_root,'tracks.'+scene_video_id+'.txt')
        print(annot_file)
        itraj_dataset = load_edinburgh(annot_file, title="Edinburgh",
                                  use_kalman=False, scene_id=scene_video_id, sampling_rate=4)  # original framerate=9
        trajs = list(itraj_dataset.get_trajectories())
        traj_datasets_per_scene.append(pd.concat([itraj_dataset.data.iloc[:, : 4], itraj_dataset.data.iloc[:, 8:9]], axis=1))

    if len(traj_datasets_per_scene)>0:
        df = pd.concat(traj_datasets_per_scene)
        df.to_pickle(month+'.pickle')
   