# TFeat shallow convolutional patch descriptor
Code for the BMVC 2016 paper [Learning local feature descriptors with triplets and shallow convolutional neural networks](http://www.iis.ee.ic.ac.uk/~vbalnt/shallow_descr/TFeat_paper.pdf)

## Network description

We provide 4 variants of the TFeat descriptor trained with combinations of different loss functions, and with and without in-triplet anchor swap. For more details check the paper. 

| network       | description   |
| ------------- |:-------------:|
| tfeat-ratio   | ratio w/out anchor swap |
| tfeat-ratio*  | ratio with anchor swap  |
| tfeat-margin  | margin w/out anchor swap|
| tfeat-margin* | margin with anchor swap |

To download the networks run the `get_nets.sh` script 

```bash
sh get_nets.sh
```
##  [New] Example usage - PyTorch

Example on how to use and train the network using Pytorch can be found [here](https://github.com/edgarriba/examples/tree/master/triplet)

## Example usage - Tensorflow

Example on how to use and train the network using Tensorflow can be found [here](https://github.com/vbalnt/tfeat/tree/master/tensorflow)

## Example usage - torch

Example on how to use the TFeat descriptor in Torch can be found [here](https://github.com/vbalnt/pnnet/blob/master/eval.lua)

## Example usage - python
[tfeat_demo.py](tfeat_demo.py) shows how to use the TFeat descriptor using python and openCV. 

To use TFeat to detect an object `object_img.png` in a video `input_video.webm` using feature point matching
```bash
python tfeat_demo.py nets/tfeat_liberty_margin_star.t7 input_video.webm object_img.png'
```

To use TFeat to just describe patches in image, run 
```bash
./extract_desciptors_from_hpatch_file.py imgs/ref.png ref.TFEAT
```

## Real-time Matching demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=S5TGfF0HLLs
" target="_blank"><img src="http://img.youtube.com/vi/S5TGfF0HLLs/0.jpg" 
alt="320" width="240" height="180" border="10" /></a>

[Real-time demo on using TFeat](https://www.youtube.com/watch?v=S5TGfF0HLLs)

## More information
More information and the full training code can be found in the [pnnet repository](https://github.com/vbalnt/pnnet)
