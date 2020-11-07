import time
import sys
import math
import torch
import torch.nn as nn

sys.path.append('/home/icalixto/tools/pretrained-models.pytorch')
import pretrainedmodels
import pretrainedmodels.utils

class PretrainedCNN(object):
    """
    Class that encompasses loading of pre-trained CNN models.
    """
    def __init__(self, pretrained_cnn):
        self.pretrained_cnn = pretrained_cnn
        self.build_load_pretrained_cnn()

    def build_load_pretrained_cnn(self):
        """
            Load a pre-trained CNN using torchvision/cadene.
            Set it into feature extraction mode.
        """
        start = time.time()
        self.load_img = pretrainedmodels.utils.LoadImage()
        image_model_name = self.pretrained_cnn
        self.model = pretrainedmodels.__dict__[image_model_name](num_classes=1000, pretrained='imagenet')
        #self.model.train()
        self.tf_img = pretrainedmodels.utils.TransformImage(self.model)
        # returns features before the application of the last linear transformation
        # in the case of a resnet152, it will be a [1, 2048] tensor
        self.model.last_linear = pretrainedmodels.utils.Identity()
        elapsed = time.time() - start
        print("Built pre-trained CNN %s in %d seconds."%(image_model_name, elapsed))

    def load_image_from_path(self, path_img):
        """ Load an image given its full path in disk into a tensor
            ready to be used in a pretrained CNN.

        Args:
            path_img    The full path to the image file on disk.
        Returns:
                        The pytorch Variable to be used in the pre-trained CNN
                        that corresponds to the image after all pre-processing.
        """
        input_img = self.load_img(path_img)
        input_tensor = self.tf_img(input_img)
        input_var = torch.autograd.Variable(input_tensor.unsqueeze(0), requires_grad=False)
        return input_var

    def get_global_features(self, input):
        """ Returns features before the application of the last linear transformation.
            In the case of a ResNet, it will be a [1, 2048] tensor."""
        return self.model(input)

    def get_local_features(self, input):
        """ Returns features before the application of the first pooling/fully-connected layer.
            In the case of a ResNet, it will be a [1, 2048, 7, 7] tensor."""
        if self.pretrained_cnn.startswith('vgg'):
            #feats = self.model.local_features(input)
            feats = self.model._features(input)
        else:
            feats = self.model.features(input)
        return feats

