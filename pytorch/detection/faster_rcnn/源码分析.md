# FasterRCNN源码分析

![fasterRCNN](图表库/fasterRCNN.png)



image输入`Backbone`，得到特征图

特征图进行RPN，得到proposals，大约2000个

proposal根据特征图和原图得到映射关系，得到特征矩阵

特征矩阵经过ROIpooling得到相同大小的特征图 `7*7`大小，flatten后经过两个全连接层FC1和FC2，然后分别经过FC3类别预测，FC4边界框回归

最后经过一些列处理



## FasterRCNN框架

### BackBone



### GeneralizedRCNNTransform

```
transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
```



### RPN

#### rpn_anchor_generator = AnchorGenerator

```
rpn_anchor_generator = AnchorGenerator(
    anchor_sizes, aspect_ratios
)
```

#### RPNHead

```
rpn_head = RPNHead(
    out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
)
```



##### init

```
self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_si
self.bbox_pred = nn.Conv2d(
    in_channels, num_anchors * 4, kernel_size=1, stride=1
)
```



##### forward

```
// 这里由rpn_anchor_generator.num_anchors_per_location 得到 num_anchors为3
bbox_pred:Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
cls_logits:Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
```



#### RegionProposalNetwork

```
rpn = RegionProposalNetwork(
    rpn_anchor_generator, rpn_head,
    rpn_fg_iou_thresh, rpn_bg_iou_thresh,
    rpn_batch_size_per_image, rpn_positive_fraction,
    rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
    score_thresh=rpn_score_thresh)
```



##### forward



```
// 五个尺度的特征图
// torch.Size([4, 256, 168, 336])
// torch.Size([4, 256, 84, 168])
// torch.Size([4, 256, 42, 84])
// torch.Size([4, 256, 21, 42])
// torch.Size([4, 256, 11, 21])
features = list(features.values())
// 五个尺度特征图经过head，每个经过两个卷积层，一个用作分类，一个用作bbox回归
// objectness: 第一个尺度torch.Size([4, 3, 168, 336]), 其他则类推
// pred_bbox_deltas: 第一个尺度torch.Size([4, 12, 168, 336]), 其他则类推
objectness, pred_bbox_deltas = self.head(features)
anchors = self.anchor_generator(images, features)
```



```
num_anchors_per_level_shape_tensors
4:torch.Size([3, 11, 21])
3:torch.Size([3, 21, 42])
2:torch.Size([3, 42, 84])
1:torch.Size([3, 84, 168])
0:torch.Size([3, 168, 336])
```



```
num_anchors_per_level
0:169344
1:42336
2:10584
3:2646
4:693
```



### ROIpooling

box_roi_pool = MultiScaleRoIAlign

```
box_roi_pool = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2)
```



### TwoMLPHead

box_head = TwoMLPHead

```
box_head = TwoMLPHead(
    out_channels * resolution ** 2,
    representation_size)
```



### FastRCNNPredictor

box_predictor = FastRCNNPredictor

```
box_predictor = FastRCNNPredictor(
    representation_size,
    num_classes)
```



### RoIHeads

roi_heads = RoIHeads

```
roi_heads = RoIHeads(
    # Box
    box_roi_pool, box_head, box_predictor,
    box_fg_iou_thresh, box_bg_iou_thresh,
    box_batch_size_per_image, box_positive_fraction,
    bbox_reg_weights,
    box_score_thresh, box_nms_thresh, box_detections_per_img)
```

