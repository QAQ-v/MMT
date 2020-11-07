# Derived from https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow/blob/master/make_flickr_dataset.py
# Also from https://github.com/elliottd/satyrid/blob/master/make_dataset.py

import numpy as np
import os
import tables
import argparse
import torch
import torch.nn as nn

from onmt.PretrainedCNNModels import PretrainedCNN


def get_cnn_features(image_list, split, batch_size, dataset_name, pretrained_cnn, pretrained_cnn_name):
    """ Function that does the actual job.

        Creates a hdf5 compressed file, iterates a list of images in minibatches,
        extracts both global and local features for these images and saves these
        features into the hdf5 file.
    """
    # create hdf5 file
    hdf5_path = "%s_%s_%s_%s" % (dataset_name, split, pretrained_cnn_name, "cnn_features.hdf5")
    hdf5_file = tables.open_file(hdf5_path, mode='w')

    # make sure feature sizes are as expected by underlying CNN architectures
    if pretrained_cnn_name.startswith('vgg'):
        global_features_size = 4096
        local_features_size  = 512 * 7 * 7
    else:
        global_features_size = 2048
        local_features_size  = 2048 * 7 * 7

    # use compression in the hdf5 file
    filters = tables.Filters(complevel=5, complib='blosc')
    # create storage for local features
    local_features_storage = hdf5_file.create_earray(hdf5_file.root, 'local_feats',
                                                     tables.Float32Atom(),
                                                     shape=(0, local_features_size),
                                                     filters=filters,
                                                     expectedrows=len(image_list))
    # create storage for global features
    global_features_storage = hdf5_file.create_earray(hdf5_file.root, 'global_feats',
                                                      tables.Float32Atom(),
                                                      shape=(0, global_features_size),
                                                      filters=filters,
                                                      expectedrows=len(image_list))
    # iterate image list in minibatches
    for start, end in zip(range(0, len(image_list)+batch_size, batch_size),
                          range(batch_size, len(image_list)+batch_size, batch_size)):
        if start%200==0:
            print("Processing %s images %d-%d / %d" 
                  % (split, start, end, len(image_list)))

        batch_list_fnames = image_list[start:end]
        batch_list = []
        # load/preprocess images for mini-batch
        for entry in batch_list_fnames:
            batch_list.append(
                    pretrained_cnn.load_image_from_path(entry))

        # create minibatch from list of variables
        # i.e., condense the list of image input variables into a mini-batch
        input_imgs_minibatch = torch.cat( batch_list, dim=0 )
        input_imgs_minibatch = input_imgs_minibatch.cuda()
        #print "input_imgs_minibatch.size(): ", input_imgs_minibatch.size()

        # forward pass using pre-trained CNN, twice for each minibatch
        lfeats = pretrained_cnn.get_local_features(input_imgs_minibatch)
        gfeats = pretrained_cnn.get_global_features(input_imgs_minibatch)
        #print("lfeats.size(): ", lfeats.size())
        #print "gfeats.size(): ", gfeats.size()

        # transpose and flatten feats to prepare for reshape
        lfeats = np.array(list(map(lambda x: x.T.flatten(), lfeats.data.cpu().numpy())))
        # flatten feature vector
        gfeats = np.array(list(map(lambda x: x.flatten(), gfeats.data.cpu().numpy())))
        local_features_storage.append(lfeats)
        global_features_storage.append(gfeats)

    print("Finished processing %d images" % len(local_features_storage))
    hdf5_file.close()


def load_fnames_into_dict(fh, split, path_to_images):
    """ Read image file names from a file into a dictionary."""
    data = dict()
    data['files'] = []

    num = 0
    # loop over the data
    for img in fh:
        if 'mscoco' in img:
            idx = img.index('#')
            img_path = img[:idx].strip()
        else:
            img_path = "%s/%s"%(path_to_images,img.strip())
        data['files'].append(img_path)
        num += 1

    print("%s: collected %d images"%(split, len(data['files'])))
    return data


def build_pretrained_cnn(pretrained_cnn_name):
    """ Uses pytorch/cadene to load pre-trained CNN. """
    cnn = PretrainedCNN(pretrained_cnn_name)
    cnn.model = cnn.model.cuda()
    return cnn


def make_dataset(args):
    cnn = build_pretrained_cnn(args.pretrained_cnn)

    # get the filenames of the images
    data = dict()
    if 'train' in args.splits:
        with open(args.train_fnames, 'r') as fh:
            data['train'] = load_fnames_into_dict(fh, 'train', args.images_path)

    if 'valid' in args.splits:
        with open(args.valid_fnames, 'r') as fh:
            data['valid'] = load_fnames_into_dict(fh, 'valid', args.images_path)

    if 'test' in args.splits:
        with open(args.test_fnames, 'r') as fh:
            data['test'] = load_fnames_into_dict(fh, 'test', args.images_path)

    for split in data:
        #files = ['%s/%s' % (args.images_path, x) for x in data[split]['files']]
        files = data[split]['files']
        get_cnn_features(files, split, args.batch_size, args.dataset_name, cnn, args.pretrained_cnn)

    print("Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the dataset bundles to train or test a model (ImgD, ImgE and ImgW).")

    parser.add_argument("--dataset_name", default="flickr30k",
                        help="""Dataset name used to create output files.""")
    parser.add_argument("--splits", default="train,valid,test",
                        help="Comma-separated list of the splits to process")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Minibatch size for processing images")
    parser.add_argument("--images_path", type=str,
                        help="Path to the directory containing the images",
                        default="/home/icalixto/resources/multi30k/images")
    parser.add_argument("--pretrained_cnn", type=str, required=True,
                        choices=['resnet50','resnet101','resnet152','fbresnet152','vgg19','vgg19_bn'],
                        help="""Name of the pre-trained CNN model available in
                        https://github.com/Cadene/pretrained-models.pytorch""")
    parser.add_argument("--train_fnames", type=str,
                        default="/home/icalixto/tools/"+
                                "lium-cvc-wmt17-mmt/data/train_images.txt",
                        help="""File containing a list with training image file names.""")
    parser.add_argument("--valid_fnames", type=str,
                        default="/home/icalixto/tools/"+
                                "lium-cvc-wmt17-mmt/data/val_images.txt",
                        help="""File containing a list with validation image file names.""")
    parser.add_argument("--test_fnames", type=str,
                        default="/home/icalixto/tools/"+
                                "lium-cvc-wmt17-mmt/data/test2016_images.txt",
                        help="""File containing a list with test image file names.""")
    parser.add_argument("--gpuid", type=int, required=True)

    arguments = parser.parse_args()

    # make sure splits are as expected
    splits = arguments.splits.split(",")
    valid_splits = ['train', 'valid', 'test']
    assert(all([s in valid_splits for s in splits])), \
        'One invalid split was found. Valid splits are: %s'%(
                valid_splits)
    arguments.splits = splits

    make_dataset(arguments)
