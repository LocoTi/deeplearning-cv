import os
import datetime

import torch
import torchvision

from anchor_utils import AnchorGenerator
from backbone_utils import resnet_fpn_backbone
from faster_rcnn import FasterRCNN, FastRCNNPredictor
from _utils import overwrite_eps
from train_utils import utils, train_eval_utils
from train_utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from coco_dataset import get_coco
import transforms

def get_model(num_classes, device="cuda:0"):

    # # https://download.pytorch.org/models/vgg16-397923af.pth
    # # 如果使用mobilenetv2的话就下载对应预训练权重并注释下面三行，接着把mobilenetv2模型对应的两行代码注释取消掉
    # vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
    # backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除feature中最后的maxpool层
    # backbone.out_channels = 512

    # # https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    # # backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.pth").features
    # # backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels

    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))

    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
    #                                                 output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
    #                                                 sampling_ratio=2)  # 采样率

    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # 加载backbone时不需要加载官方模型
    backbone = resnet_fpn_backbone('resnet50', pretrained=False, trainable_layers=3)
    model = FasterRCNN(backbone, num_classes=91)  # 加载官方模型，这里num_classes不能修改

    # 加载fasterrcnn_resnet50_fpn官方模型
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict = torch.load("C:/Users/lixiao/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    overwrite_eps(model, 0.0)

    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # 修改predictor

    return model

def main(args):
    # utils.init_distributed_mode(args)
    # print(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    # # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # # Data loading code
    print("Loading data")
    
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    coco_root = args.data_path
    # check root
    if os.path.exists(coco_root) is False:
        raise FileNotFoundError("coco dose not in path:'{}'.".format(coco_root))
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])  # number of workers
    print('Using %g dataloader workers' % nw)
    
    val_dataset = get_coco(coco_root, "val", data_transform["val"])
    
    print("Creating data loaders")
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # else:
    #     train_sampler = torch.utils.data.RandomSampler(dataset)
    #     test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    # if args.aspect_ratio_group_factor >= 0:
    #     group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
    #     train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    # else:
    #     train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    print("Creating model")
    model = get_model(num_classes=args.num_classes + 1)
    model.to(device)
    print(model)

    model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    train_loss = []
    learning_rate = []
    val_map = []

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    if args.test_only:
        train_eval_utils.evaluate(model, val_data_set_loader, device=device)
        return

    for epoch in range(args.start_epoch, args.epochs):
        metric_logger = train_eval_utils.train_one_epoch(model, optimizer, val_data_set_loader, device, epoch, args.print_freq)
        
        mean_loss, lr = metric_logger["mloss"], metric_logger["lr"]

        train_loss.append(mean_loss)
        learning_rate.append(lr)
        
        lr_scheduler.step()
        
        # evaluate after every epoch
        coco_info = train_eval_utils.evaluate(model, val_data_set_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss, lr]]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'resnet-fpn-model-{}.pth'.format(epoch)))


    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)

if __name__ == "__main__":
    import argparse

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='D:/work/data/Python/coco2017', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=90, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.002, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.StepLR的参数
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[8, 10], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练的batch size(如果内存/GPU显存充裕，建议设置更大)
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument('--distributed', type=bool, default=False, help='if train on multi GPU or not')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)