import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        

    def finetuned(self):
        # 1 - Finetuning from a pretrained model
        # Let’s suppose that you want to start from a model pre-trained on COCO and want to finetune it for your particular classes. 
        # Here is a possible way of doing it:

        # load a model pre-trained on COCO
        coco = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # replace the classifier with a new one, that has num_classes which is user-defined
        coco_num_classes = 2  # 1 class (person) + background

        # get number of input features for the classifier
        in_features = coco.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        coco.roi_heads.box_predictor = FastRCNNPredictor(in_features, coco_num_classes)

        return coco
    
    def backbone(self):
        # 2 - Modifying the model to add a different backbone

        # load a pre-trained model for classification and return only the features
        model = torchvision.models.mobilenet_v2(weights="DEFAULT").features

        # ``FasterRCNN`` needs to know the number of output channels in a backbone. 
        # For mobilenet_v2, it's 1280 so we need to add it here
        model.out_channels = 1280

        # let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios. 
        # We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # let's define what are the feature maps that we will use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling. if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an ``OrderedDict[Tensor]``,
        # and in ``featmap_names`` you can choose which feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        # put the pieces together inside a Faster-RCNN model
        return FasterRCNN(
            model,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    def get_model_instance_segmentation(self, num_classes):
        # 3 - we want to finetune from a pre-trained model, given that our dataset is very small, so we will be following approach number 1.
        # Here we want to also compute the instance segmentation masks, so we will be using Mask R-CNN:
        
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )

        return model