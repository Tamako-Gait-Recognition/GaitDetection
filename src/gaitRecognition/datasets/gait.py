import numpy as np
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    '''
    Args:
        data_list_path(string): 骨骼点数据地址
        transform:  对数据集的转化
    '''
    def _filename_to_target(self,filepath:str):
        # raise NotImplemented()
        #filename是带扩展名的文件名

        #sequence-id属于视频名，如1-1表示id为1的人的第1个视频
        sequence_id,filename = filepath.split("/")[-2:]

        subject_id, sequence_num = sequence_id.split("-")
        framenum = filename.split(".")[0] #帧序号(int类型的)
        return (
                (int(subject_id),int(sequence_num)),
                int(framenum)
            )

    def __init__(self,
                 data_list_path:str,
                 posedatalength: int,  #骨骼点数据长度
                 sequence_length=60,
                 transform = None,
                 ):
        super(PoseDataset,self).__init__() #调用父类构造函数
        self.data_list = np.loadtxt(data_list_path,skiprows=1,dtype=str)  #读入骨骼点数据

        self.transform = transform
        self.sequence_length = sequence_length

        self.poseInfo_dict = {}  #骨骼信息字典
        '''
        暂且写出这样的文件分割，之后再讨论文件名格式
        '''

        for idx,row in enumerate(self.data_list):
            row = row.split(",")  #读入骨骼信息csv并根据逗号分割
            target,frame_num = self._filename_to_target(row[0])

            if target not in self.poseInfo_dict:
                self.poseInfo_dict[target] = {}

            #在当前项目架构中骨骼数据csv一行是52列，其中第一列是文件路径相关信息
            #因此再不满足骨骼信息长度（骨骼信息有缺失），直接跳过
            if len(row[1:]) != posedatalength:
                continue

            try:
                self.poseInfo_dict[target][frame_num] = np.array(
                    row[1:], dtype=np.float32
                ).reshape((-1, 3))
            except ValueError:
                continue
        '''
        后面可能要处理少帧的问题
        '''

        self.targets = list(self.poseInfo_dict.keys()) #list({人数id:帧id})
        self.data = list(self.poseInfo_dict.values())  #list(对应的骨骼点数据)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        '''
        参数 index(int):代表人的id

        返回值
            tuple(骨骼数据，(人id，帧序号))
        '''
        target = self.targets[index]
        data = np.stack(list(self.data[index].values()))
        if self.transform is not None:
            data = self.transform(data)
        return data,target

    def get_num_people(self):
        '''
        :return: 返回人id
        '''
        if type(self.targets[0]) == int:
            humans = set(self.targets)
        else:
            humans = set([target[0] for target in self.targets])
        num_humans = len(humans)
        return num_humans

class PoseQueryDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 transform=None):
        super().__init__()
        self.transform = transform