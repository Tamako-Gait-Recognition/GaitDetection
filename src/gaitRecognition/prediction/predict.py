import torch
from torchvision import transforms
from gaitRecognition.datasets.augmentation import *
from gaitRecognition.datasets.gait import *
from gaitRecognition.models.SpatialTransformerTemporalConv import SpatialTransformerTemporalConv,SpatioTemporalTransformer
from gaitRecognition.common import *
from gaitRecognition.sampler import *
from gaitRecognition.losses import *
from pytorch_metric_learning import miners, losses
from gaitRecognition.prediction.evaluate import *
import os
def predict():
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    device = torch.device("cuda")
    opt = parse_option()
    transform = transforms.Compose(
        [
            MirrorPoses(opt.mirror_probability),
            FlipSequence(opt.flip_probability),
            RandomSelectSequence(opt.sequence_length),
            PointNoise(std=opt.point_noise_std),
            JointNoise(std=opt.joint_noise_std),
            joint_drop if opt.joint_drop == 'single' else lambda x:x,
            remove_conf(enable=opt.rm_conf),
            normalize_width,
            ToTensor()
        ],
    )

    if opt.loss_func == 'supcon':
        transform = TwoNoiseTransform(transform)
    val_transform = transforms.Compose(
        [
            SelectSequenceCenter(opt.sequence_length),
            remove_conf(enable=opt.rm_conf),
            normalize_width,
            ToTensor()
        ]
    )

    dataset = PoseDataset(
        opt.train_data_path,
        #等待修改为变量，骨骼点数据长度
        posedatalength=51,
        sequence_length=opt.sequence_length,
        transform=transform
    )

    dataset_valid = PoseDataset(
        opt.valid_data_path,
        posedatalength=51,
        sequence_length=opt.sequence_length,
        transform=ThreeCenterSequenceTransform(
            transform=val_transform,
            sequence_length=opt.sequence_length
        )
    )

    if opt.balance_sampler:
        person_number = opt.train_id

    persons_id = []
    for t in dataset.targets:
        persons_id.append(t[0])

    persons_id = torch.tensor(persons_id)

    _sampler = BalancedBatchSampler(
        labels=persons_id,n_classes=person_number,n_samples=opt.sampler_num_sample
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=opt.num_workers,
        pin_memory=True,
        batch_sampler=_sampler,
    )
    #验证集的dataloader
    val_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=opt.batch_size_validation,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    model = None
    if opt.model_type == "spatialtransformer_temporalconv":
        model = SpatialTransformerTemporalConv(
            num_frame=opt.sequence_length, in_chans=2 if opt.rm_conf else 3, spatial_embed_dim=opt.embedding_spatial_size, out_dim=opt.embedding_layer_size, num_joints=17, kernel_frame=opt.kernel_frame)
    elif opt.model_type == "spatiotemporal_transformer":
        model = SpatioTemporalTransformer(
            num_frame=opt.sequence_length, in_chans=2 if opt.rm_conf else 3, spatial_embed_dim=opt.embedding_spatial_size, out_dim=opt.embedding_layer_size, num_joints=17)
    elif opt.model_type == "gaitgraph":
        model = get_model_resgcn()

    load_checkpoint(model, opt)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if opt.loss_func == 'supcon':
        loss_func = SupConLoss(temperature=0.004, base_temperature=0.004)
    else:
        loss_func = losses.TripletMarginLoss(margin=0.01)
    if opt.cuda:
        model.cuda()
        loss_func.cuda()

    evaluate(val_loader,model,opt.evaluation_fn)


