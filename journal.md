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

## Apr. 9, 2018

I trained the model for 19 epochs, with an initial learning rate of 0.01 and using the dev-decay method with a patience of 1; the training error seemed to plateau at about 0.85, regardless of how much the learning rate was reduced, and the best validation error was at just under 1.1. From this attempt, a few things are clear:

* the model is not high-capacity enough to overfit the training set, which is usually required for the network to fit will;

* given that this model was large enough to fit Imagenet fairly well, it seems clear that the goal function itself (i.e., the combination of data and labels) is very difficult to learn. 

Before I move on, I should note some additional important points. 

* I used momentum of 0.9 in my SGD. In that paper I mentioned yesterday, pure SGD performed better than SGD with momentum. 

* When using `ReduceLROnPlateau`, I was checking whether the *validation loss* was plateauing, rather than the *validation accuracy*. This is an important distinction, and it might affect the final accuracy of the model (the issues regarding loss, mentioned above, are not related). 

The next thing to do (this weekend):

* add a command line option for loading a (resnet) model from a file;

* write a function for evaluating a model's *accuracy*, instead of the model's *loss*;

* evaluate the saved model's accuracy on the training and validation set to get a sense of how the loss is corresponding to the accuracy (since the problem is so difficult, a higher loss may be acceptable in that it still yields near-100% accuracy);

* qualitatively explore the mistakes the model is making (what classes are correlated? Does it *make sense* that they would be correlated, or is it completely abnormal? I predict that it will make sense, but I must exercise due diligence in these matters and investigate properly). 

## Apr. 14, 2018

I wrote and tested the accuracy calculation functions, and streamlined the testing code to reduce code duplication. 

I integrated the new accuracy calculation functions into the main script, including sections involving training and validation, the script output, the metric tracking, and the step function for the `ReduceLROnPlateau` learning rate scheduler. 

Next step is adding an evaluation mode. A related issue is saving the hyperparameters with the model. 

I added an evaluation mode, and am ignoring how I save the hyperparameters. In doing that, I added the command line parameter allowing me to load a model from a file. I also evaluated the model's accuracy on the validation set: 0.67 validation accuracy, and 1.17 validation loss. On the one hand, these results seem abysmal in comparison to Imagenet or other image dataset accuracies and losses. On the other hand, this is a pretrained model, which means the following: since it did not do well, I have either (a) failed to optimize the hyperparameters well enough or (b) the problem is just *hard* to solve. The evidence supporting (b) is that, even though the validation loss is a high value of 1.17 (indicating that the probability of the validation predictions being entirely correct is e^-1.17, or approximately 0.31), the accuracy seems relatively high at nearly 70% (far, *far* better than 1/128 random-model performance). HOWEVER, I don't have good reference numbers to back this up. 

[This post and the corresponding answers](https://stats.stackexchange.com/questions/258166/good-accuracy-despite-high-loss-value) suggest that the relationship between accuracy and loss is not necessarily inverse--but that's just two random guys on Stats Stack Exchange. 

The next thing to do is implement saving of the metrics along with the model; additionally, I want a --name option in the CLI, and save all things relating to the model's run to the folder `models/--name`. This way, we don't have to worry about saving everything in one object. 

Once I have this completed, I will rerun the training again and watch the accuracy *and* the loss to observe the behavior. 

I added the `models/--name` option, and saved all the hyperparameters I thought were relevant. 

I'm re-running my first experiment. Changes from the first experiment, though:

* I'm using no momentum term;

* I'm stopping training entirely if the learning rate goes below 10^-8; and

* the learning rate scheduler is stepping based on the validation accuracy, rather than the validation loss. 

A thought about this whole project, from looking at a Kaggle Kernel today: the categories are unclear. There are no labels for them. Some of them are obvious enough (glasses out of which you drink, vs. cups out of which you drink, vs. pants), but others are not entirely clear. There may be some semantic issues with the labels--i.e., it's tough to generalize because you can't generalize well in the first place. 

I wonder if attentional models would do better at paying attention to the details?

## Apr. 20, 2018

I couldn't train the model, because I had to do taxes, and I didn't have support for resuming from a pause in training. 

I will start the training--however, I have an idea. The `resnet18` model is clearly not a good-enough architecture for this problem. Since the model bias is the largest factor in determining good performance, I think that a larger model will yield greater performance. To this end...

* I will use the deeper `resnet34` model instead. 

* This means my batch size will be cut in half. This also means I will need to perform another hyperparameter search. 

* On a related note, the `ReduceLROnPlateau` learning rate scheduler will reduce the learning rate by a factor of 0.5. Compared to reducing by a factor of 0.1, this should help by keeping the learning rate as high as possible while also decreasing the learning rate significantly. 

* This time, I won't have to worry about a potentially-bad momentum value, since I'm not using momentum; this is nice. 

* I will perform my learning rate search starting with 1 and dividing by 2 for four iterations: 1, 0.5, 0.25, 0.125, and 0.0625. If any edge value gets the lowest loss, I will keep searching in that direction until the learning rate increases. For each attempted learning rate, the validation loss will be evaluated at the end of one epoch through the dataset; this is the loss for that learning rate. 

Actually, I have a different learning rate search procedure; the goal of this procedure is to save time, since an entire epoch is expensive. I start at a high learning rate of 1.0 with just 1000 samples, and multiply by 0.5 as required until my training loss converges to near-0. Once I obtain this maximum learning rate, I immediately jump to training on the entire training set; however, instead of the maximum learning rate I obtained before, I use `(maximum learning rate) * (1000 / (size of training set))`. The rationale behind this is that, when multiplying the training set size by a factor `x`, it is equivalent to multiplying the batch size by `1/x`. Since the batch size and learning rate have a linear relationship (according to the "ImageNet in an Hour" paper), you simply multiply the learning rate by `1/x` to get the correct learning rate for the whole training set. 

I performed this learning rate search for this dataset, and obtained a maximum learning rate of 0.5 for 1000 samples with a batch size of 64. Since the whole training set is approximately 192000 samples, I have now started training with a learning rate of 0.5/192, or approximately 0.002604 (this is the exact value I used). We will see the results in approximately three days. Peace out man. 

## Apr. 21, 2018

Upon futher reflection, it occurs to me to go *bigger*. Since resnet-34 only took up 4 GB of space on my GPU, I want to go up to resnet-50, with the largest batch size possible. My minimum batch size will be 32. Let's see if I can get that high. I got to batch size 64 successfully with resnet50. Success!

I tested five different learning rates on one epoch of 16000 images: 1.0, 0.5, 0.25, 0.125, and 0.0625. 0.25 was the highest learning rate that performed the same as 0.125 and 0.0625. Now, to scale up to 192k images, I will multiply the learning rate by 16000/192000 to get a final learning rate of approximately 0.083. Now I can run a full experiment, with 41 epochs over the dataset, in order to perform my final experiment. 

Aaaaaaaand I shouldn't do that yet. I should figure out how well this model can fit those 16000 images in a full training run. Then, once it's clear that the validation loss is somewhat decent, I can double the size of the training set and re-run to see if more data yields any improvement on the validation loss. Now, I will run with 16000 images and observe the results. 

Turns out the resnet50 model still can't overfit the training data (loss about 0.5-0.6 on training set). I've run the hyperparameter search on resnet101 and come up with 0.25 as the best learning rate. I will now complete a full training run on resnet101 with learning rate 0.25, and report the results. 

After training it for some time, it definitely yields a better training loss than resnet50 (after 13 epochs the training loss is 0.4--already better than 0.5-0.6, which was the training loss for resnet50), indicating that it is, indeed, able to fit the data better. However, I suspect that even this model will not be good enough. From the MIT paper "Toward Robust Neural Networks Against Adversarial Examples" (or something like that; it's in another repository of mine), I know that wider networks do better against adversarial examples; further, [this](https://arxiv.org/abs/1605.07146v3) paper on Wide Residual Networks suggests that wider is much, much better than deeper. For these reasons, I will try the Wide Residual Network from that paper if this model does not completely overfit the 16000-image subset of the training data. 

## Apr. 22, 2018

The training loss seemed to plateau at 0.4, showing that it cannot overfit the data. I strongly believe that width is going to play an important role in obtaining good performance on this dataset. Based on this intuition, my time would be better spent implementing a Wide Residual Network than running another experiment on a pretrained DenseNet. 

OH MY GOD. I just realized I was only training the final linear layer of each of these Resnet architectures. FUCK. OF COURSE I WAS GETTING TERRIBLE PERFORMANCE. Aw man. Training the entire network, I should be getting much better results. 

First: back down to Resnet18, and doing a batch size test. The largest batch size I can use for resnet18 is 320. Now, let's do a hyperparameter search. The search shows that 0.25 is the largest learning rate which can be used without diverging and losing performance. Now, let's run a full training run on our 16000 images and see if the model can overfit. 

The model is unable to overfit (loss 2.5 on training set). Time to move up to resnet50. Run the batch size tests. Largest batch size possible for resnet50 is 80. Now run the hyperparameter search. The best learning rate appears to be 0.125. Now I will run this run to see if it can overfit the data. 

It did *not* overfit the data. The loss plateaued at a high value again (around 0.7). Clearly, these networks are not enough. I need a shallower, wider network. Next time: using a batch size of 32, multiply the width of a ResNet18 network (number of feature maps in each layer) by a larger and larger constant until no memory is left. Then, we will train the network and see if it can overfit the training data (or, at least, the order-16000 subset of the training images). 'Til next time. 

## Apr. 29, 2018

I looked into applying the Net2WiderNet transform from [this](https://arxiv.org/abs/1511.05641) old 2015 paper; although it is easy in principle, it's not straightforward to apply this transformation to architectures with skip connections. An easier way to obtain a wider network is to use DenseNet and vary the growth factor while training from scratch. 

Additionally, I looked at [this](https://arxiv.org/abs/1804.07612) paper about small-batch training, which advises the use of small batch sizes for better performance. I will try a constant batch size of 32, which was among the best-performing batch sizes on ImageNet (anywhere between 8 and 64 seemed to work fine, given the best learning rate for that batch size). I will test learning rates $2^0, 2^{-1}, ..., 2^{-12}$, which is the set of learning rates tested in the paper, and will select the one which yields the best validation loss after a single epoch. 

Before I complete this, though, I need to adjust my primary script so that it resets the value of `metrics` every time the `main(...)` function is called. This will prevent the problem I observed earlier, during hyperparameter searches, where the loss histories were appended to each other after each run. In addition, I need to fix the lack of log output in the hyperparameter search. 

I fixed the main script to reset the value of `metrics` (I took out threading entirely). I still couldn't fix the lack of log output during hyperparameter searching. However, I found the optimal value for the learning rate to be $2^{-10}$. Now I'll run a full experiment with it: 100 epochs, or when the learning rate drops below $10^{-8}$. Now let's run our experiment. 

## Apr. 30, 2018

EXPERIMENT RUN, AND IT OVERFIT THE TRAINING SET, BY GOD! FANTASTIC! A problem I need to fix: when do I stop training? Basically, I'm having issues with the learning rate scheduler that I need to fix. But that's for tonight. I need to go to work. Peace out!

Aaaaaand I'm back! I've got one hour, so let's make this count. First thing to fix: early stopping issue. Fixed it! Had to directly access `optimizer.param_groups[0]['lr']` to see whether the learning rate was below my chosen threshold. 

Running with 32000 examples, DenseNet can *still* overfit the data--like, hardcore overfit. A couple notes:

* the validation performance is 2% better with 32000 examples so far, but still not spectacular (77% accuracy on validation, with 99.8% accuracy on training);

* the training loss is very unstable (I think this is because I'm not limiting the upper and lower probability limits that I assign to each class; as a result, confidence in wrong answers becomes catastrophic);

* the training accuracy is somewhat unstable (i.e., it seems to fluctuate close to 100% accuracy).

The next steps are to (1) scale up to the full training set, and (2) add data augmentation (random flipping/cropping/rotating, random color augmentation). The purpose of these goals is to decrease the generalization error that the model is currently experiencing. 

## May 5, 2018

After training on the full dataset, it (DenseNet, with same learning rate chosen from earlier) achieved 85% validation accuracy and nearly-100% training accuracy. This is much better than 77% accuracy, which I achieved before. I suspect that I'll be able to achieve even better accuracy with (a) augmented test data, and (b) a larger network (which may be *required* after I add the data augmentation). 

Reading the paper [Learning to Compose Domain-Specific Transformations for Data Augmentation](https://arxiv.org/abs/1709.01643), I learned about two common assumptions in data augmentation: (a) that the augmented image will be from the same data distribution as the original image, and (b) that the augmented image will have the same class label as the original image. 

Assumption (b) is likely to be violated only when dealing with either (1) non-object classification datasets (i.e., patterns, microscopic images, medical images, rock formations, etc.), or (2) object classification datasets in which multiple objects from different categories may be present in a single image (even though only one class label is correct). Case (2) is a problem because some transformations may occlude the object corresponding to the true label, while also *not* occluding some object corresponding to an incorrect label. 

Our dataset corresponds to case (2) above, and so constraint (b) applies. 

In order to make sure these constraints hold for a given transformation sequence (stochastic or otherwise), I should perform a set of experiments on myself. First, I must learn the dataset myself. Second, I must iterate over (choosing the transformation sequence) and (evaluating whether it preserves the class). 

For learning the data:

1. Get human-understandable labels for each category. 

2. Organize into a tree of category types (tree leaves are classes) to simplify classification by a human. 

3. Train myself to classify the dataset in order to know what is in the dataset and what is not. 

For choosing the transformation sequence:

1. Heuristically (or randomly, or by some other process) choose a transformation sequence. 

2. Classify images (myself). Calculate accuracy. If accuracy drops significantly, go back to step #1. 

Ideally, we'd go into even more depth: balancing the class distribution to be uniform, analyzing per-class human misclassification of transformed images (vs per-class accuracy of originals), and probably more of which I haven't thought. 

Of course, this is *WAY TOO MUCH WORK* for me to do on the weekends. So, instead, I'll do something simpler. For each random transformation, I'll pick a few random samples, perform a given transformation, pick the transformation parameters providing the largest distortion which maintains the object in question within the image, and then combine all transformations (performing a final check to see whether they all work together without changing the class). 

BUT EVEN BEFORE THAT I need to figure out the class labels for every class. Okay, maybe just do all this for *one* class. If generalizationg is still bad, then maybe one class may be augmented too much (so it changes class/is outside the distribution). 

First thing's first: find the class label for a single class. This means I need to (a) load the images for a given class, and (b) display them next to each other. 

From [this](https://www.kaggle.com/andrewrib/visualizing-imaterialist-data) Kaggle Kernel, we know that category 40 is definitely chairs. So, we'll look at the chair images with different transform values for the ColorJitter transform (with RandomHorizontalFlip & RandomRotation & Zoom & RandomCrop occurring before, and Grayscale occurring after that with some small probability--say, 0.1). 

## May 6, 2018

I wrote a script to inspect the images generated by the preprocessing transformation, inspected the images to determine the largest distortion values I could use for preprocessing which also kept the image's true label from changing, added a(n optional) `tqdm` progress bar to the training and validating functions, and started a hyperparameter search for the augmented data. Also, I switched to using a deeper network (densenet161 instead of 121). 

## May 7, 2018

The hyperparameter search shows that learning rate 2^-9 works well (so does 2^-10, but higher a learning rate will, hopefully, learn slightly faster). HOWEVER, the model barely gets 75% accuracy on the training data, so I'd rather not scale up to the full dataset. I still need to overfit the training data. To that end, I'll attempt to use an even *deeper* network. Can my GPU memory even handle a deeper network? Anyway, this is for later tonight. Peace out man!

After thinking about the problem some more today, I came up with the following:

> Data augmentation caused a *severe* drop in performance. 

> First: maybe it just takes more time than a single epoch to achieve good training performance. Second: maybe one of the augmentations is really hindering learning. 

> Testing the first case takes more time, but requires less effort. It involves training, with the best learning rate from the hyperparameter search I've already finished, on the full dataset and with full augmentation. 

> Testing the second case takes less time, but involves potentially more effort. It involves removing all augmentation and then reintroducing it, one transformation at a time, to find out which augmentation is causing the performance drop; each time, train for two epochs on a 16k-sample subset of the training data, with the best learning rate from the hyperparameter search. 

> I'd rather test the second case before the first, since that seems more likely. That's what I'll do tonight. 

So, as I mentioned above, I will strip out all augmentations before adding them back in. 

After removing all augmentations except for random flipping, the network was only achieving a training accuracy of about 75% after two epochs. However, I considered that it might simply be taking longer to converge--and it was! After 8 epochs, the training accuracy was 99%. This means that the larger model just took longer to converge. It also suggests that densenet121 might have converged, had I given it more time to do so. Regardless, I've started a training run of densenet161 on fully-augmented training data. 

If it converges, then the model--being larger than densenet121, and training on random data augmentations--takes longer to converge, and I simply didn't give the model (densenet161) enough time to converge in the hyperparameter search. If it does not converge, then my initial assumption--that the dataset augmentations are causing the problem--is correct instead. 

Now it's time to sleep. GO SLEEP. 

## May 8, 2018

IT OVERFIT! WOOO! So it was just a case of slower convergence due to data augmentations and a larger model! This leads me to wonder whether resnet121 would actually have converged, had I given it much more time to do so. Hmmm. A problem for another day, however. For now, we know that resnet161 converges on the training set. With this in mind, I have started running resnet161 on the whole dataset, with the same settings as in the hyperparameter search and as in the training run with a 16000-sample subset. I expect to see a validation accuracy of around 90% (optimistically), given that a full training run with no augmentation on densenet121 yielded 85% validation accuracy. 

If I still achieve only 85% accuracy on the validation set, I suspect that removing the grayscale transform will improve the model's performance, given that there are likely no grayscale images in the validation dataset. Of course, this would need to be evaluated on a per-class basis. Speaking of, I should write a script to evaluate the model's accuracy and performance on a per-class basis. This would give me more information to work off of when trying to make improvements to the model; this might be especially useful for figuring out which data augmentations are causing issues. 

## May 13, 2018

Thoughts from earlier this week:

> Densenet161 with full augmentation got 97% accuracy on training, but 85% accuracy on validation. This is same as densenet121 with no augmentation! Only simsilarity is center crop; that may be what's preventing generalization. 

> Best thing to do is visualize errors on validation set. Since they got same validation performance, I'll first look at densenet121 with no augmentation at all and evaluate what errors it makes. Then I can do the same for densenet161 and see if similar or different errors are occurring. 

> If center crop is causing the issue, then likely errors will occur when identifying tall or wide objects that occupy most of the image. If it is a tougher semantics problem (i.e., multiple objects in the image bug object of focus (image center) is the true label), then I expect to see errors with scenery images (as in, full-kitchen picture for a stool ad). 

To this, I'll add: that any issues which I assume are caused by center crop issue should appear in both architectures. 

What is a simple code modification to obtain the images which are causing errors? I can copy the evaluation function (in my `training.py` module) and make a new, modified version of it which returns the avg error, along with a tuple of `(image.cpu(), label)` pairs. I don't need the original files; just the images themselves will be fine. (Later, I will need a way to get the original files in order to make submissions--but, for now, I don't need this.)

I made the code modification. Now, in the `evaluate()` function in the `finetune_resnet.py` module (still need to change that name), I need to display a few things about the errors: class distribution, and a random sample of the error images (as well as the class number overlayed). 

After displaying all of these things, I also need to know the class names in order to interpret the results. Using [this](https://www.kaggle.com/andrewrib/visualizing-imaterialist-data) kernel by Andrew, I can get display a large sample from each category. From that, here are my labels for each category:

1. outdoor lounge chair
2. thermos
3. office chair
4. living/dining room chair
5. stove pot with lid
6. rocking chair
7. TV
8. dining room tables
9. swiss army knife/multi-tool
10. bed
11. hot/cold water dispensers
12. lamp
13. "bed chair" (small bench/chair-ish thing to put at the end of your bed)
14. mirror
15. dining room chair
16. rice cooker
17. no-legged cushion chair-ish
18. a psychologist's patient sofa
19. coffee table
20. a glass (for drinking water/etc.)

Okay, I can't do this forever. I'm done for this weekend. 

## May 14, 2018

21. living room couch
22. living room chair
23. some sort of chair?? (omg too many chair categories)
24. wardrobe
25. shelf/rack
26. PLASTIC chair

I think I get the categories. It's a combination of furniture type and material type. Having said that, let me go back and edit some of these categories. Hmmm. Except, in category 1, there does not seem to be a consistent material. Maybe it's either/or? Like, it could be a furniture category, or it could be a furniture + material? Maybe it's the style of chair, then? Not entirely sure. 

Regardless, these are *difficult* categories. I will have to rethink my straightforward neural net approach. Maybe a decision tree of neural networks for particularly difficult categories? 

Hmmm. Regardless, let's just try using random crop with desnenet121, using the same learning rate as before, but adding the non-color-changing data augmentations. The color-changing augmentations may actually hinder generalization if the color is important to the style (bold vs soft colors, etc.). 
