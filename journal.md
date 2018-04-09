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

## Mar. 19, 2018

I've decided it's worth the upload, because it downloads everything into directories so that it is useable for PyTorch. Aaaaaaaand I realized two things: (1) I can't upload a Kernel without running it (pretty terrible, really), and (2) it's probably not worth the effort. 

From looking at [this](https://www.kaggle.com/iezepov/this-is-a-text-classification-competition) Kernel--specifically, the comments--I now know that using the URLs is not allowed in the competition. Good to know. 

From looking at [this](https://www.kaggle.com/codename007/simple-exploration-notebook-furniture) Kernel, I've learned a few relevant things:

* there are no missing URLs (even though, from my own experience, some URLs are dead);

* the class distribution is relatively, though not exactly, even.

From looking at [this](https://www.kaggle.com/andrewrib/visualizing-imaterialist-data) Kernel, I've learned some more things:

* the image distribution is heavily skewed (the previous Kernel was misleading, because it grouped multiple categories into the same bin);

* the functions below are cool;
  ```
  from IPython.core.display import HTML 
  from ipywidgets import interact
  from IPython.display import display
  ```

* the images are product images, which often feature the product front and center--even if they are in their natural environment. 

Even though the categories are diverse, the *shapes* of the objects are quite similar. I wonder how they differ between categories? The next step in basic data exploration is to assign a name to each of the 128 features, and do some more data exploration myself. 

Actually, the *easiest* thing to do is to get a pretrained PyTorch model and run that on the dataset. That's this weekend's goal. Meanwhile, I should look at shape detection literature. 

Side note: the shape only gives you a hint. The context of the image will probably be important. 

## Mar. 20, 2018

Want to know the complexity of the data. How to measure complexity? Easy test: how simple of a model can you fit while getting good performance?

* Logistic regression on progressive layers of Imagenet-trained model's outputs; how deep before performance is reasonable?

* How well does t-SNE do in separating the images?

* Is the center of the image always the product? Center crop vs. random crop. Maybe show with attention-based mechanism...but probably too complicated. Easier just to show effectiveness by taking severe center crop (like, only take the center X% of the least-magnitude dimension, where X varies from 10 to 100, and then train on those reduced images; alternatively, blurring the image beyond certain radius from center--but that's just a complicated version of the previous idea, so just take the center X%). Prediction: shrinking center crop will improve prediction in some classes, but it will reduce the ability to distinguish between similar shapes with different contexts. 

* How much do you lose by switching to grayscale?

* Try random shallow-convnet outputs with different models: random forests, linear/logistic regression, SVM. Can they do well?

* Shape of objects seems like a really good indicator. Can you strip everything else--i.e., take only the edges of objects from an image--and still get good results?

--Need background of image, too, for more ambiguous stuff (cup or waterbottle? First has kitchen background, second has camping background). How much do you need this? Can you do well with *just* the background image--i.e., *remove* the center of the image?

Two interesting experiments from these thoughts: take center X% of center-cropped image and see performance, from X=10 to 100; then take *outer* X% of center crop and see performance, from X=10 to 100. 

## Mar. 21, 2018

I've decided that, this weekend, I want to put out a Kernel in which I perform the experiment above. In order to do this, I'll need a model that's easy to use and gets decent performance. To this end, I'll set up the framework--not for test submissions, but for training a model and evaluating on the validation data. To that end, tonight I'm going to choose the model, create the image transform, and get the linear outputs (that is, before logistic function is performed) from the model. 

Model chosen: resnet34, because it gets good top-5 performance (~8% error, competitive with the best) while also being small enough to fit on my laptop's GPU. 

The transform is done, and it was easy. 

How do I get the "guts" of the function? Specifically, I'm trying to get the 2nd-to-last layer's output in resnet34. I can access the variables which refer to each layer. From this, I could build a class which uses each of the layer variables from a pretrained instance. However...this is just inheritance, but without calling it inheritance! I just need to subclass resnet34; in the constructor for the subclass, I will call my superclass's constructor with `pretrained=True`; in the `forward(self, x)` member function, I will construct everything as normal except that I will leave out the final fully-connected layer. This will give me the 2nd-to-last layer's output. Let's test this. 

DAMNIT. `resnet18` is a *function*, not a *class*, so I can't inherit from `resnet18`--which means I can't inherit from a pretrained model. So it seems like my initial idea was the better option. However, I'm exhausted and I'll do that tomorrow. 

## Mar. 25, 2018

Let's get our second-to-last layer output from resnet18. 

The idea works! Use an actual instance of `ResNet` (`resnet18` or `resnet34`) and use its layers directly in my own class! I wrote the unit tests. 

I learned that I can use the statement form `assert bool_expression, "wny it'd be false"` to include an error message in my assert statements. That is *hella* useful. 

Inner functions only have *read* access to the enclosing scope! This makes it *way* easier to conceive of how closures would be implemented for the purposes of returning functions from functions. 

Decorators are functions which return wrapped functions. That's all. We can use the `functools.wraps` decorator on the definition of the inner (i.e. wrapped) function in order to preserve the function name, docstring, and args (default and non-default); `wraps` even preserves the type annotations! Beautiful. 

Use inner classes for encapsulation. More Python programmers should do this; it would have helped a lot. HOWEVER, if you're already encapsulating only one primary class per file then you should not use inner classes for helper classes which the user should access. 

Use `@property`'s instead of regular variables for anything you intend to allow a user to set. 

Enums are new in Python 3! Woohoo! Use enums; then, for checking whether the value is allowed in a setter, use a single type check on the enum class instead of checking whether the value is in a set. It's cheaper, and simpler!

After that fun learning session, it's time to move on to the train/eval loop. I'd like to get tensorboard working, as well, so I can monitor the training progress of the network with cool visualizations. 

Turns out that there are some version issues that I don't want to deal with. Let's try a more official installation sequence for tensorflow and tensorboard--maybe from the official TensorFlow website. Maybe that will solve any issues I have. 

Nope! First, the wrong version of CUDA was installed--so I had to install TensorFlow v1.4.1 instead of the latest TensorFlow (because CUDA 9.0 caused problems for my GTX980M GPU). Then, once I had finished that, I found that it requires CUDNN; for 1.4.1, it requires CUDNN 6.0. That requires going to the NVIDIA website, downloading the files, and copying them into `/usr/local/cuda-8.0/include` or `.../cuda-8.0/lib64` as appropriate (file structure of archive indicates where files should go). After this, I got *another* error. I didn't even look at it. It's not worth installing TensorFlow; I've already spent 30 minutes attempting it. A different weekend, maybe. 

Maybe. 

If I can't do that, then the next step is to write the training loop. 

Finished writing the training loop. Also figured out how to simplify implementation of resnet copy with different last layer. Gonna run it through tonight. At 1500 iterations, we have a full epoch through the dataset. The resnet paper (original mentions taking 64k iterations with a batch size of 128, which makes 8 million (approximately) images--which is 8 times the size of the labelled portion of ImageNet. Since our dataset is smaller, we can go through 8 epochs of our own dataset, or 192000*8/128 or approximately 12000 iterations. That should be plenty. Here's to hoping. 

## Mar. 27, 2018

I tried finetuning imagenet. After awhile, I found that the loss was super noisy and plateaued around 0.7-1.0. That was with Adam (batch size of 128), though, rather than with SGD (dividing by 10 when reaching a plateau, max of 64k iterations with batch size of 256)--which was the original method. 

Training a model from scratch (again with Adam) did not work well at all; the loss didn't go down nearly as quickly as the finetuned model. 

Training a model from scratch with SGD (as in the original paper) did not work well. The loss initially *increased* due to instability in the training. Not sure what the source of it was. 

To figure out what it was, I tried step #0 of [this](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607?gi=279f1037a5a3) guide for debugging neural network training--that is, I reduced the number of training examples to 500 and retrained. It didn't overfit quickly, which is concerning. Is the data just really hard to learn? I verified that the images were, in fact, legitimate. 

I tried going back to using Adam, but even that didn't work well. Maybe using Adam with 0.001 instead of 0.0001 learning rate? I.e., maybe the learning rate was too small?

Turns out that `shuffle=True` in the `DataLoader` constructor causes the training data to be reshuffled on *every epoch*. It's cool, but it also means I can't shuffle once and then train on the first few entries in that subset. 

The solution is to take away the `--no-shuffle` option. That way, we shuffle every time, but we can use the `RandomSubsetSampler` as the sampler if we need to restrict the training data. Tada!

Increasing the size of the training data to just 8100 prevents the model from converging quickly. Let's try smaller sizes. $2^7 * 8 = 1024$, so let's try `--training-subset 1000` with `--batch-size 128`. Even a training set as small as 1024 doesn't converge quickly! Let's try even smaller, with 512--so with `--training-subset 500`. 

Fun fact: I wasn't actually using the `lr_scheduler.ReduceLROnPlateau` object at all. I have to call `step(loss)` on it after every epoch, where `loss` is the validation loss. So that's cool. That's probably why `Adam` was working when `SGD` was not working. Also, I discovered that the `Adam` optimizer has a default learning rate of `0.001`, which is different than the `0.0001` that I thought it was. 

Increasing the initial learning rate to 0.1 for `Adam` did the trick! Now it converges with 128, 256, and 512 examples! Let's go back up to 1024 and see what's up. BUT BEFORE THAT we need to discuss the loss of a uniform distribution over the categories. That loss is approximately 1/128. The cross entropy loss for classification evaluates to `E[-log(p_hat)]`, where `p_hat` is the probability assigned to the correct class by the model, and where the expectation is over the training examples. Thus, if the model assigns 1/128 to each class, then we would see `-log(1/128)`, or about 2.1 loss. OKAY. Now we can continue. 

While that model's running, let's go over some notes about what we've seen so far. 
* Increasing the number of examples also increases the time it takes to fit those examples. E.g., it took about 150 iterations to converge for 128 examples. For 256 examples, it took about 300-400. For 512 examples, it took about 1000. For 1024 examples, we will see how long it takes. 
* Increasing the number of examples also increases the initial instability in the batch training loss. The length of this period of instability initially increases with the number of examples , due to the batch gradient not being the same as the gradient of the true loss. (However, I predict that, past some large-enough number of examples, the length of this initial period of instability should remain constant.)
* Increasing the number of examples increases variance in the batch training loss throughout training. 

Lesson learned: FOLLOW THE GUIDE. Choose an existing model, overfit on `batch_size` number of images. Use this overfitting to debug your network: increase your initial learning rate until just before it starts diverging, display the images you're training on to make sure your program is generating images correctly (are they actually images, and not just blank? In the small batch size, is your iterator iterating correctly?). 

Back to the training! It finished. It converged to around 0.1-0.2 loss after about 2500-3000 iterations. Awesome! The next thing to do is calculate a per-channel mean and standard deviation. I need to use multiprocessing to do it, because otherwise it will be *slow as hell*. 

## Apr. 1, 2017

I need to get the channel-wise mean and channel-wise pixel quantity for each image. To get it for a single image, I have a function `channel_sum_and_size(image)`. That works just fine. The problem is adding multiprocessing into the mix. I need to limit the number of jobs I can finish at one time. 

Aaaaaaand (hours later, late at night) I'm done! Here are the results, for posterity:

```
mean = [0.6837, 0.6461, 0.6158]
stdev = [0.2970, 0.3102, 0.3271]
```

OH MY GOD THAT TOOK FOREVER. I am so done with that. UGH. 

Now we insert the normalization as part of the transform in our original code. 

Interestingly, the convergence rate did not change significantly. At about 3000 iterations, the training starts to diverge. 

The next thing to do is set up the training as normal: complete an epoch over the training data, and then compute the validation set loss, with the learning rate decreasing whenever the validation loss does not decrease. 

Before-bed note: learning rate maximum seems to be 0.01 before it diverges. 

## Apr. 7, 2018

On my earlier project, the regular `Adam` optimizer implementation was at least 10x slower than in Tensorflow. Maybe if I use the `SparseAdam` optimizer then it will converge more quickly. Let's try that. 

`SparseAdam` apparently only works for sparse tensors (I guess there is a specific datatype which is different from a regular tensor). 

I tried optimizing a pretrained network with Adam, learning rate 0.01, on 1024 samples; it converged very quickly to ~0.05 training error (about 400 steps). I tried optimizing the same thing on 2048 samples and 4096 samples; in both cases, it converged quickly. However, when moving to 2^13 samples with the same setup, I had to decrease the learning rate to 0.001. It slowly converged (0.1 - 0.2 training error after 3000 iterations), but it converged nonetheless. 

After doing all of that, I found the paper ["The Marginal Value of Adaptive Gradient Methods in Machine Learning"](https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf) from NIPS 2017, which suggests two things:

* use SGD instead of Adam, and use validation-based decay (i.e., evaluate on validation set after every epoch; when current validation error is not the minimum so far, decrease the learning rate by constant factor);

* if you use Adam, decay the learning rate (probably using validation-based decay). 

This provides a clear path forward, at least. What I now need to do is implement the training loop and validation loop separately. Here are some relevant implementation details:

* make sure to take the average error over the *whole* validation set, rather than the average of batch errors on the validation set;
* one method for training, the other for validating. 

Implemented the single-epoch training function. 

## Apr. 8, 2018

Now I need to implement the validation function. It should take the data, the network, and a loss function. The loss function must sum, not average, the losses in a batch. 

I made the validation function. Also fixed both train and validate functions to use cuda as an option. 

I have made progress since the previous paragraph was written:

* I rewrote the main script in order to use the `train` and `validate` functions;

* I made the `Stopwatch` class instances into context managers for more convenient and sensible syntax;

* I organized my code into three threads--the training thread, the command thread, and the main thread--and set the training thread as a daemon and make the main thread perform a join with the command thread (this makes it so that, if the command thread exits because I type in `exit`, then the main thread will finish its join and stop, causing the training thread to stop instantly (since it's a daemon));

* after making sure the code worked properly, I tested the code on three learning rates: 0.1, 0.01, and 0.001. The learning rate 0.01 was the best among them, so I will start with that and use the `ReduceLROnPlateau` learning rate scheduler, stepping after each epoch and passing the validation loss. 

Two things yet to do before I run the experiment: find the optimal patience value to use for `ReduceLROnPlateau`, and only save the model corresponding to the best validation score (at least, I think that's the thing to do). Let's check that last paper, ["The Marginal Value of Adaptive Gradient Methods in Machine Learning"](https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf), that we dug up. 

Looking at the paper, the "dev-decay" method is what they claim is best, and it has a patience of 1, so we'll use a patience of 1 as well. Now let's add the method for saving the model with the best validation error. 
