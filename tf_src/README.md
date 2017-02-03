# TFeat shallow convolutional patch descriptor

This example implements the paperBMVC 2016 paper [Learning local feature descriptors with triplets and shallow convolutional neural networks](http://www.iis.ee.ic.ac.uk/~vbalnt/shallow_descr/TFeat_paper.pdf) in [Tensorflow](https://www.tensorflow.org).

The implementation is very close to the Torch implementation [pnnet.torch](https://github.com/vbalnt/pnnet).

After the initialization a log file under the name of `flags.txt` is genereated with all the user parameters.
We assume that you have downloaded and extracted the dataset to `data_dir`.

After every epoch, models are saved to: `log_dir/logname/model-train_name-%`.

```
usage: train_tfeat.py [-h] [--image_size IMAGE_SIZE]
                      [--num_channels NUM_CHANNELS] [--activation ACTIVATION]
                      [--train_name TRAIN_NAME] [--test_name TEST_NAME]
                      [--num_triplets NUM_TRIPLETS] [--batch_size BATCH_SIZE]
                      [--num_epochs NUM_EPOCHS] [--margin MARGIN]
                      [--learning_rate LEARNING_RATE]
                      [--weight_decay WEIGHT_DECAY] [--optimizer OPTIMIZER]
                      [--data_dir DATA_DIR] [--log_dir LOG_DIR]
                      [--log_name LOG_NAME] [--gpu_id GPU_ID] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --image_size IMAGE_SIZE
                        The size of the images to process (default: 32)
  --num_channels NUM_CHANNELS
                        The number of channels per image (default: 1)
  --activation ACTIVATION
                        The default activation function during the training.
                        Supported: tanh, relu, relu6 (default: tanh)
  --train_name TRAIN_NAME
                        The default dataset name for training. Supported:
                        notredame, liberty, yosemite. (default: notredame)
  --test_name TEST_NAME
                        The default dataset name for testing. Supported:
                        notredame, liberty, yosemite. (default: liberty)
  --num_triplets NUM_TRIPLETS
                        The number of triplets to generate (default: 1280000)
  --batch_size BATCH_SIZE
                        The size of the mini-batch (default: 128)
  --num_epochs NUM_EPOCHS
                        The number of training epochs (default: 60)
  --margin MARGIN       The margin value for the loss function (default: 1.0)
  --learning_rate LEARNING_RATE
                        The initial learning rate (default: 1e-4)
  --weight_decay WEIGHT_DECAY
                        The weight decay (default: 1e-4)
  --optimizer OPTIMIZER
                        The optimizer to use during the training. Supported:
                        Momentum, Adam. (default: Momentum)
  --data_dir DATA_DIR   The path to the patches dataset. (default:
                        /tmp/patches_dataset)
  --log_dir LOG_DIR     The path to the logs directory (default:
                        /tmp/tensorboard_log)
  --log_name LOG_NAME   The name for logging (default: triplet_tanh
  --gpu_id GPU_ID       The default GPU id to use (default: 1)
  --seed SEED           The random seed (default: 666)
```
