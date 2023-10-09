import os
import csv
from typing import Optional,Tuple

import numpy as np
import cv2
from poseEstimation.visualization.utils import chunhua_style as color_style
from poseEstimation.visualization.utils import map_joint_dict
'''
参数：
VideoListPath中是视频存放地址（文件夹）
VideoName是想获取的视频的视频名，无扩展名

返回值：
若不存在则是None，存在则是(帧率，扩展名，(视频宽度，视频高度))
'''
def fun():
    return (1,(2,3))
def getVideoInfo(VideoListPath:str,VideoName:str) -> Optional[Tuple[int,str,Tuple[int,int]]]:
    VideoFiles = os.listdir(VideoListPath)
    VideoFiles = [item for item in VideoFiles if os.path.isfile(os.path.join(VideoListPath,item))]
    result = None
    for item in VideoFiles:
        if item.startswith(VideoName):
            cap = cv2.VideoCapture(os.path.join(VideoListPath,item))
            #获取帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            #获取原视频宽度、高度
            Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            suffix = "." + item.split(".")[-1]
            cap.release()
            result = (fps,suffix,(Width,Height))
            break
    return result

'''
poseDataPath 指骨骼点数据csv在路径
ImgListPath 指对应的帧图片路径（全部）
'''
def Visualiztion(PoseDataPath : str,ImgListPath:str,VideoListPath:str,OutputPath:str):
    id = dict()  # 每个视频存有多少张图片
    img_file = dict()  # 每张图片
    frame_data = dict()  # 每张图片的文件名
    base_path = os.getcwd()  # 获得当前工作目录(默认是当前py文件所处文件夹)

    with open(PoseDataPath) as file:
        reader = csv.reader(file)
        header = next(reader)  #存下第一行并跳过
        front = ""  #用于存储原视频名，用于后续创建文件
        filename_list = []  #图片名列表
        img_data_list = []  #骨骼点数据列表
        cnt = 0
        for row in reader:
            #获取视频名和帧名
            splitResult = row[0].split('/')
            sequence_id = splitResult[-2]
            frame = splitResult[-1]

            frame_num = int(frame[:-4])    #去掉后缀，将数字串转化为数字int
            data = np.array(row[1:], dtype=np.float32).reshape((-1, 3))
            #下面是判断是否是属于同一视频的帧，不是就分开存储
            if front == sequence_id or front == "":
                if front == "":
                    front = sequence_id
                filename_list.append(frame)  #frame是对于每个视频中的各个视频帧的文件名
                img_data_list.append(data)   #data是对于每个视频中的各个视频帧对应的骨骼数据，用于后续绘图
                #print(filename_list)
                cnt += 1
            else:
                #一旦进入就表示是不同的视频了，先把前面视频的保存到字典
                id[front] = cnt
                img_file[front] = filename_list
                frame_data[front] = img_data_list
                filename_list = []
                img_data_list = []
                #开始存储新的帧数据
                cnt = 0
                filename_list.append(frame)
                img_data_list.append(data)
                #更新视频文件名
                front = sequence_id
        #对最后的一个视频进行存储
        id[front] = cnt
        img_file[front] = filename_list
        frame_data[front] = img_data_list
    #output_size = (720,1280)

    for s_id,num in id.items():
        cnt = 0
        '''
        VideoWriter()的参数：
        参数1：要保存的文件路径
        参数2：指定编码器
        参数3：要保存的视频的帧率
        参数4：要保存的文件的画面尺寸（可选）
        '''
        list_img_file = img_file[s_id]  #图片名列表
        list_frame_data = frame_data[s_id] #骨骼点数据列表
        fps,suffix,VideoSize = getVideoInfo(VideoListPath,s_id)

        #图像的尺寸必须与视频的尺寸一致，视频才可以正常显示
        out = cv2.VideoWriter(os.path.join(OutputPath, s_id + suffix),cv2.VideoWriter_fourcc('X','V','I','D'),fps,
                              VideoSize)

        for i in range(len(list_img_file)):
            frame = np.zeros((VideoSize[0],VideoSize[1],3),dtype=np.uint8)
            filename = list_img_file[i]
            data = list_frame_data[i]
            #print(os.path.join(base_path,'data/frame',s_id,filename))
            # img = cv2.imread(os.path.join(base_path, '../../data/frame', s_id, filename))
            img = cv2.imread(os.path.join(ImgListPath,s_id,filename))
            height,width,_ = img.shape

            #以下为绘图程序
            if data is not None:
                joints_dict = map_joint_dict(data)
                for k, link_pair in enumerate(color_style.link_pairs):
                    cv2.line(
                        img,
                        (joints_dict[link_pair[0]][0], joints_dict[link_pair[0]][1]),
                        (joints_dict[link_pair[1]][0], joints_dict[link_pair[1]][1]),
                        color=np.array(link_pair[2]) * 255
                    )
            pos_x = cnt % 4
            pos_y = cnt // 4

            #frame[pos_y*height:(pos_y+1)*height, pos_x*width:(pos_x+1)*width] = img   #这行可能是仅对视频的人物进行保存

            out.write(img)
            cnt += 1
        out.release()
        cv2.destroyAllWindows()

