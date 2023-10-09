import sys
import time
import torch
from gaitRecognition.prediction.utils import AverageMeter
import numpy as np
import pandas as pd

def evaluation_function(embeddings):
    # print(embeddings)
    persons_id = [k[0] for k in embeddings.keys()]
    '''
    for t in embeddings.keys():
        if len(persons_id) == 0 or t[0] != persons_id[-1]:
            persons_id.append(t[0])
    '''

    #persons_id = list(set(t[0] for t in embeddings.keys()))
    person_pre_frame = {}
    for id in persons_id:
        person_pre_frame[id] = {
            k:v for (k,v) in embeddings.items() if k[0] == id
        }
    persons_embaddings = np.array(list(embeddings.values()))
    globalCorrect = 0
    globaltotal = 0
    personcorrectDict = {}
    persontotalDict = {}
    #定义中间变量
    #...

    for (target,embedding) in embeddings.items():
        subject_id,_ = target

        persons_embaddings_norm = persons_embaddings / \
            np.linalg.norm(persons_embaddings,ord=2,
                           axis=1,keepdims=True)

        embedding_norm = embedding / \
            np.linalg.norm(embedding,ord=2,keepdims=True)

        distance = 1 - persons_embaddings_norm @ embedding_norm

        min_pos = np.argmin(distance)   #最近距离

        min_dis_id = persons_id[int(min_pos)]  #最近距离对应的id
        #暂时先统计总体准确率和每个人的准确率
        if subject_id not in persontotalDict:
            persontotalDict[subject_id] = 1
            personcorrectDict[subject_id] = 0
        else:
            persontotalDict[subject_id] += 1

        if min_dis_id == subject_id:
            globalCorrect += 1
            personcorrectDict[subject_id] += 1

        globaltotal += 1
    # print(persons_id)
    globalaccuracy = globalCorrect / globaltotal
    person_accuracy_list = []
    for id in set(persons_id):
        person_accuracy_list.append(personcorrectDict[id] / persontotalDict[id])
    person_accuracy_list = [float(x) for x in person_accuracy_list] + [globalaccuracy]
    dataframe = pd.DataFrame(
        person_accuracy_list,
        index=list(set(persons_id)) + ['mean']
    )
    print(dataframe)
    return globalaccuracy,person_accuracy_list


def evaluate(data_loader,model,evaluation_fn,gpu = False):
    model.eval()
    batch_time = AverageMeter()
    use_flip = True

    with torch.no_grad():
        end = time.time()
        embeddings = dict()
        for index,(posepoints,target) in enumerate(data_loader):
            # dimensions = [len(sublist) for sublist in posepoints]
            # print(dimensions)
            is_3seq = False  #判断是否为3维序列
            if isinstance(posepoints,list):
                is_3seq = True
            if is_3seq and not use_flip:
                raise ValueError(
                    "Average 3 Seq without using flip is not supported"
                )
            if use_flip:
                if isinstance(posepoints,list):
                    posesize = posepoints[0].shape[0]
                    data_flipped0 = torch.flip(posepoints[0],dims=[1])
                    data_flipped1 = torch.flip(posepoints[1],dims=[1])
                    data_flipped2 = torch.flip(posepoints[2],dims=[1])
                    posepoints = torch.cat(
                        [posepoints[0],data_flipped0,posepoints[1],data_flipped1,posepoints[2],data_flipped2],dim=0)
                else:
                    posesize = posesize.shape[0]
                    data_flipped = torch.flip(posepoints,dims=[1])
                    posepoints = torch.cat([posepoints,data_flipped],dim=0)

            #gpu可用就移动到gpu进行运算
            if torch.cuda.is_available():
                posepoints = posepoints.cuda(non_blocking=True)

            output = model(posepoints)

            if use_flip:
                if is_3seq:
                    split6 = torch.split(output,[posesize] * 6,dim=0)
                output = torch.mean(torch.stack(split6),dim=0)

            else:
                split1,split2 = torch.split(output,[posesize] * 2,dim=0)
                output = torch.mean(torch.stack([split1,split2]),dim=0)

            for i in range(output.shape[0]):
                sequence = tuple(
                    int(t[i]) if type(t[i]) is torch.Tensor else t[i] for t in target
                )

                if gpu:
                    embeddings[sequence] = output[i]
                else:
                    embeddings[sequence] = output[i].cpu().numpy()

            # batch_time.update(time.time() - end())
            end = time.time()

            #每10次输出一次数据
            if index % 10 == 0:
                print(
                    f"Test: [{index}/{len(data_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                )
                sys.stdout.flush()
    return evaluation_fn(embeddings)