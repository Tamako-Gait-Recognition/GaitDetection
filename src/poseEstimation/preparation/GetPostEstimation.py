import subprocess
from prepare_detection import detection
from prepare_pose_estimation import pose_estimation
from poseEstimation.visualization.visual import Visualiztion
import os
# !!!骨骼点那里少一张图片，之后去检查原因
def GetPostEstimation(bash:str,videoPath:str,targetPath:str,isVisual : bool):
    '''
    #将视频分割成图片
    result = subprocess.run(['bash','../../../data/extract_frames.sh','../../../data/video','../../../data/frames'])
    #预测姿态，提取骨骼点
    detection("", "../../../data/frames/all_frames.csv", "../../../data/frames/detections.csv")
    pose_estimation("", "../../../data/frames/detections.csv", "../../../data/frames/pose_coco.csv")
    '''
    result = subprocess.run(['bash',bash,videoPath,targetPath])
    detection(os.path.join(targetPath,"all_frames.csv"),os.path.join(targetPath,"detections.csv"))
    pose_estimation("",os.path.join(targetPath,"detections.csv"),os.path.join(targetPath,"pose_coco.csv"))



if __name__ == '__main__':
    #GetPostEstimation(False)
    #test

    Visualiztion(r"E:\code file pycharm\GaitDetection\data\frames\pose_coco.csv"
                 ,r"E:\code file pycharm\GaitDetection\data\frames"
                 ,r"E:\code file pycharm\GaitDetection\data\video"
                 ,r"E:\code file pycharm\GaitDetection\data\output")



