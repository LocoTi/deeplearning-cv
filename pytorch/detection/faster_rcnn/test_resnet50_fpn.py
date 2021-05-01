import os
import datetime

import torch


from faster_rcnn import fasterrcnn_resnet50_fpn

from backbone_utils import mobilenet_backbone, resnet_fpn_backbone
from faster_rcnn import FasterRCNN

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(pretrained=True)
print(model)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

print(params)
checkpoint = torch.load("C:/Users/lixiao/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", map_location=device)
model.load_state_dict(checkpoint['model'])

print(params)


# mobilenet_v2
# import torchvision
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# backbone.out_channels = 1280

# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))

# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
#                                                 output_size=7,
#                                                 sampling_ratio=2)

# # put the pieces together inside a FasterRCNN model
# model = FasterRCNN(backbone,
#                    num_classes=2,
#                    rpn_anchor_generator=anchor_generator,
#                    box_roi_pool=roi_pooler)
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)

# resnet50_fpn
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# # For training
# images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
# labels = torch.randint(1, 91, (4, 11))
# images = list(image for image in images)
# targets = []
# for i in range(len(images)):
#     d = {}
#     d['boxes'] = boxes[i]
#     d['labels'] = labels[i]
#     targets.append(d)
# output = model(images, targets)
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)

# optionally, if you want to export the model to ONNX:
# torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

