# Large-Scale Bird Sound Classification using Convolutional Neural Networks
By [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Thomas Wilhelm-Stein](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en), [Hussein Hussein](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en), [Holger Klinck](http://www.birds.cornell.edu/page.aspx?pid=1735&id=489), [Danny Kowerko](https://www.tu-chemnitz.de/informatik/mc/staff.php.en), [Marc Ritter](), and [Maximilian Eibl](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en)

## Introduction
Code repo for our submission to the LifeCLEF bird identification task BirdCLEF2017. This is a refined version of our original code described in the working notes. We added comments and removed some of the boilerplate code. If you have any questions or problems running the scripts, don't hesitate to contact us.

Contact:  [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Technische Universität Chemnitz](https://www.tu-chemnitz.de/index.html.en), [Media Informatics](https://www.tu-chemnitz.de/informatik/Medieninformatik/index.php.en)

E-Mail: stefan.kahl@informatik.tu-chemnitz.de

This project is licensed under the terms of the MIT license.

Please cite the paper in your publications if it helps your research.

```
@article{kahl2017large,
  title={Large-Scale Bird Sound Classification using Convolutional Neural Networks},
  author={Kahl, Stefan and Wilhelm-Stein, Thomas and Hussein, Hussein and Klinck, Holger and 
  Kowerko, Danny and Ritter, Marc and Eibl, Maximilian},
  journal={Working notes of CLEF},
  year={2017}
}
```

<b>You can download our working notes here:</b> [TUCMI BirdCLEF Working Notes PDF](https://box.tu-chemnitz.de/index.php/s/RCXS6jHr2f6jypc) <i>(Unpublished draft version)</i>

## Installation
This is a Thenao/Lasagne implementation in Python for the identification of hundreds of bird species based on their vocalizations. This code is tested using Ubuntu 14.04 LTS but should work with other distributions as well.

First, you need to install Python 2.7 and the CUDA-Toolkit for GPU acceleration. After that, you can clone the project and run the Python package tool PIP to install most of the relevant dependencies:

```
git clone https://github.com/kahst/BirdCLEF2017.git
cd BirdCLEF2017
sudo pip install –r requirements.txt
```

We use OpenCV for image processing; you can install the cv2 package for Python running this command:

```
sudo apt-get install python-opencv
```

Finally, you need to install Theano and Lasagne:
```
sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

You should follow the Lasagne installation instructions for more details: 
http://lasagne.readthedocs.io/en/latest/user/installation.html

## Training
In order to reproduce our 2017 submission, you need to download the BirdCLEF2017 training data. Nonetheless, the code will work with all other bird recordings obtained from sources like Xeno-Canto, eBird or others. 

### Dataset
The training script uses subfolders as class names and you should provide following directory structure:

```
dataset   
¦
+---species1
¦   ¦   file011.wav
¦   ¦   file012.wav
¦   ¦   ...
¦   
+---species2
¦   ¦   file021.wav
¦   ¦   file022.wav
¦   ¦   ...
¦    
+---...
```
 
 For the BirdCLEF2017 training data, you can use the script <b>birdCLEF_sort_data.py</b> providing the paths of WAV and XML directories. We used a 10% local validation split for testing. You can separate files for testing from the training data by running the script  <b>birdCLEF_validation_split.py</b> and specifiying the path of your sorted dataset.

### Extracting Spectrograms
We decided to use magnitude spectrograms with a resolution of 512x256 pixels, which represent five-second chunks of audio signal. You can generate spectrograms for your sorted dataset with the script <b>birdCLEF_spec.py</b>. You can switch to different settings for the spectrograms or change the heuristic which separates bird sounds from noise by editing the file.

Extracting spectrograms might take a while. Eventually, you should end up with a directory containing subfolders named after bird species, which we will use as class names during training. 

<b>Note:</b> You need to remove the <i>“noise”</i> folder containing rejected spectrograms without bird sounds from the training data. 

### Training a Model
You can train your own model using either the BirdCLEF2017 training data or your own sound recordings. All you need are spectrograms of the recordings. Before training, you should review the following settings, which you can find in the <b>birdCLEF_train.py</b> file:

- `DATASET_PATH` containing the spectrograms (subfolders as class names)

- `NOISE_PATH` containing noise samples (you can download the samples we used [here](https://box.tu-chemnitz.de/index.php/s/SYRXElhPd6QtA0u) or select your own from the noise folder with rejected spectrograms)

- `MAX_SAMPLES_PER_CLASS` limiting the number of spectrograms per bird species (Default = 1500)

- `VAL_SPLIT` which defines the amount of spectrograms in percent you like to use for monitoring the training process (Default = 0.05)

- `MULTI_LABEL` for softmax outputs (False) or sigmoid outputs (True); Activates batch augmentation (see working notes for details)

- `IM_SIZE` defining the size of input images, spectrograms will be scaled accordingly (Default = 512 x256 pixels)

- `IM_AUGMENTATION` selecting different techniques for dataset augmentation

- `MODEL_TYPE` being either 1, 2 or 3 depending on the model architecture you like to train (see working notes for details)

- `BATCH_SIZE` defining the number of images per batch; reduce batch size to fit model in less GPU memory (Default = 128)

- `LEARNING_RATE` for scheduling the learning rate; use `LR_DESCENT = True` for linear interpolation and `LR_DESCENT = False` for steps

- `PRETRAINED_MODEL` if you want to use a pickle file of a previously trained model; set `LOAD_OUTPUT_LAYER = False` if model output size differs (you can download a pre-trained model [here](https://box.tu-chemnitz.de/index.php/s/iPUsAA94KPtWaVf))

- `SNAPSHOT_EPOCHS` in order to continuously save model snapshots; select `[-1]` to save after every epoch; the best model params will be saved automatically after training

There are a lot more options - most should be self-explanatory. If you have any questions regarding the settings or the training process, feel free to contact us. 

<b>Note:</b> In order to keep results reproducible with fixed random seeds, you need to update your <i>.theanorc</i> file with the following lines:

```
[dnn.conv]
algo_bwd_filer=deterministic
algo_bwd_data=deterministic
```

Depending on your GPU, training will take while. Training with all 940k specs from the BirdCLEF2017 training data takes 1-2 hours per epoch on a NVIDIA P6000 and 2-4 hours on a NVIDIA TitanX depending on the type of model architecture used.

## Evaluation
After training, you can test models and evaluate them on your local validation split. Therefore, you need to adjust the settings in <b>birdCLEF_evaluate.py</b> to match your model hyperparameters. The most important settings are:

- `IM_SIZE`, `MODEL_TYPE`, `MULTI_LABEL` and `TRAINED_MODEL` where you specify the pickle file of your pre-trained model and the corresponding model architecture

- `BATCH_SIZE` to fit forward pass into GPU memory and most importantly to generate timestamps for soundscapes if set to 1
- `SPEC_LENGTH` and `SPEC_OVERLAP` to test different numbers of specs per sound file; when processing soundscapes, you should set `SPEC_OVERLAP = 0`

- `INCLUDE_BG_SPECIES` is rather BirdCLEF specific and lets you decide if you want to evaluate background species, too. (Note: Not all background species listed in the xml files are relevant, therefore you should set `ONLY_BG_SPECIES_FROM_CLASSES = True`)

- You can save predictions for ensemble pooling if you specify an `EVAL_ID` and set `SAVE_PREDICTION = True`. Next time you start the script, you can load this prediction and it will be merged with the prediction of the current model

If you use any other than the BirdCLEF trainig data, you will have to adjust your ground truth before you can evaluate. You should do this by implementing the `getGroundThruth()` function of the script.

This repo will not suffice for real-world applications, but you should be able to adapt the testing script to your specific needs.

We will keep this repo updated and will provide suitable testing functionality in the future.
