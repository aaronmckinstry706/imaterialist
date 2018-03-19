# Journal for iMaterialist Competition

## Mar. 16, 2018

I'm setting out some goals for this weekend. First: download the URLs. Second: write a Python script to download all the individual URLs and organize it so that it is easily consumable by Pytorch. Off to download the URLs. 

Downloaded the URLs, and loaded them using Python's `json` module. Next step: writing script to download all the images with their corresponding labels. 

## Mar. 17, 2018

Now to write the script to download all the images. As per the competition website, the data has an "images" section and an "annotations" section. The "images" section is a list of {"url": ..., "image\_id": ...} dicts. The "annotations" section is a list of {"image\_id": ..., "label\_id": ...} dicts. So, I will loop through the urls and download each image into different folders for each class. The folders will be named for their label\_id. 

(Side note: use [`urllib.request`](https://docs.python.org/3.0/library/urllib.request.html) module to download images.)

Hmmm. I'd like to put it into a format that is easily digestible by Pytorch. Let's check how Pytorch loads images. As per [this](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html) tutorial on loading data, we need to override the `Dataset` class' `__len__()` and `__getitem__(i: int)` methods. What should `__getitem__(i: int)` return: an `(image, label)` tuple, or just the `image`? I see! It can return whatever it wants. The `DataLoader` class instance will just return a batch of whatever the `Dataset`'s `__getitem__(i: int)` method returns. Coolio! So...what does this mean for how I store the images?

* Structure of training data directory will be like so: `/data/training/<label>/<id>.jpg`. Here, `/data` denotes the `data` directory in the root directory of the project. 

* Structure of the testing directory: `/data/testing/<id>.jpg`. This is because no test labels are given. 

* We will have separate `Dataset` subclasses for the testing and training data. `TestingDataset`'s `__getitem__(i: int)` method will return a `TrainingSample` instance; a `TrainingSample` will be a subclass of `NamedTuple` with the fields `id: str`, `image: PIL.Image.Image` (obtained via `PIL.Image.open(filename: str)`), and `label: int`. 

Scratch all of that. The module `torchvision.datasets` already has a class `ImageFolder` which does everything I need it to do, except for getting the image id. I am glad I found this, rather than implementing everything myself. As a side note: I should do the image augmentation transform in the data loading (i.e., pass the transform to the `ImageFolder` instance), so that I can take advantage of the easy multiprocessing in Pytorch's `DataLoader` class. Meanwhile, via the [source code for `ImageFolder` and `DataFolder`](https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py), I know that:

* we can go from class label to class index and back with the `class_to_idx` and `classes` attributes; and

* the `default_loader` for the `ImageFolder` class uses PIL's `Image.open` function, and thus the image returned is a PIL image (meaning we can use the normal transforms on it).

So that's it! Problems solved. Now let's download those images. 

I need to get the current directory of a Python notebook. How do I do this? `os.getcwd()`. Apparently, when Pycharm runs the notebook server, it runs the server from the project's root directory. This makes things muuuuch easier. Also, when structuring my project, I can use the project root instead as the typical `/src` directory, since the project root is implicitly added to the Python path whenever I'd run anything in or out of Pycharm, anyway. 

## Mar. 18, 2018

Today, I wrote the script for downloading the images, basing it off of an existing script on Kaggle. I added a few more options and fine-tuned it to perform quickly on my computer. 

Next weekend, the goal is to figure out how to get the image annotation text, instead of just the label ids. (See "It's actually a text competition!" post on Kaggle; that's why I think it may be important.)
