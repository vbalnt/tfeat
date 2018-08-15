# TFeat shallow convolutional patch descriptor
Code for the BMVC 2016 paper [Learning local feature descriptors with triplets and shallow convolutional neural networks](http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf)

## Pre-trained models
We provide the following pre-trained models:

| network name      | model link                                                        | training dataset   |
| -------------     | :-------------:                                                   | -----:             |
| `tfeat-liberty`   | [tfeat-liberty.params](./pretrained-models/tfeat-liberty.params)  | liberty (UBC)      |
| `tfeat-yosemite`  | [tfeat-yosemite.params](./pretrained-models/tfeat-yosemite.params) | yosemite (UBC)     |
| `tfeat-notredame` | [tfeat-notredame.params](./pretrained-models/tfeat-notredame.params) | notredame (UBC)    |
| `tfeat-ubc`       | coming soon...                                                    | all UBC            |
| `tfeat-hpatches`  | coming soon...                                                    | HPatches (split A) |
| `tfeat-all`       | coming soon...                                                    | All the above      |


## Quick start guide
To run `TFeat` on a tensor of patches:

```python
tfeat = tfeat_model.TNet()
net_name = 'tfeat-liberty'
models_path = 'pretrained-models'
net_name = 'tfeat-liberty'
tfeat.load_state_dict(torch.load(os.path.join(models_path,net_name+".params")))
tfeat.cuda()
tfeat.eval()

x = torch.rand(10,1,32,32).cuda()
descrs = tfeat(x)
print(descrs.size())

#torch.Size([10, 128])
```

Note that no normalisation is needed for the input patches, 
it is done internally inside the network. 

## Testing `TFeat`: Examples (WIP)
We provide an `ipython` notebook that shows how to load and use 
the pre-trained networks. We also provide the following examples:

- extracting descriptors from image patches
- matching two images using `openCV`
- matching two images using `vlfeat`

For the testing example code, check [tfeat-test notebook](tfeat-test.ipynb)

## Re-training `TFeat`
We provide an `ipython` notebook with examples on how to train
`TFeat`.  Training can either use the `UBC` datasets `Liberty,
Notredame, Yosemite`, the `HPatches` dataset, and combinations 
of all the datasets. 

For the training code, check [tfeat-train notebook](tfeat-train.ipynb)
