# DeepVOT - Automatic Measurement of Voice Onset Time and Prevoicing using Recurrent Neural Networks
Voice onset time (VOT) is defined as the time difference between the onset of the burst and the onset of voicing. 
When voicing begins preceding the burst, the stop is called prevoiced, and the VOT is negative. 
When voicing begins following the burst the VOT is positive. 
While most of the work on automatic measurement of VOT has focused on positive VOT mostly evident in American English, in many languages the VOT can be negative. 
We propose an algorithm that estimates if the stop is prevoiced, and measures either positive or negative VOT, respectively.  More specifically, the input to the algorithm is a speech segment of an arbitrary length containing a single stop consonant, and the output is the time of the burst onset, the duration of the burst, and the time of the prevoicing onset with a confidence. Manually labeled data is used to train a recurrent neural network that can model the dynamic temporal behavior of the input signal, and outputs the events' onset and duration. Results suggest that the proposed algorithm is superior to the current state-of-the-art both in terms of the VOT measurement and in terms of prevoicing detection.

## Content
The repository contains code for VOT and prevoicing measurement, feature extraction and visualization tools.
 - `back_end folder`: contains the training algorithms, it can be used for training the model on new datasets or using different features.
 - `front_end folder`: contains the features extraction algorithm, it can be used for configuring different parameters for the feature extraction or just for visualization.
 - `post_process folder`: contains the post processing algorithms for extracting the measurements from the network probability distribution
 - `visualization folder`: contains features visualization tools.
 - `run_all folder`: contains the scripts and models to run the code end-to-end.

## Installation
The code runs on MacOSX only.
### Dependencies
The code uses the following dependencies:
 - Torch7 with RNN package
```bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh 

# On Linux with bash
source ~/.bashrc
# On Linux with zsh
source ~/.zshrc
# On OSX or in Linux with none of the above.
source ~/.profile

# For rnn package installation
luarocks install rnn
```
 - [Python (2.7) + Numpy] (https://penandpants.com/2012/02/24/install-python/)
 - For the visualization tools: [Matplotlib] (https://penandpants.com/2012/02/24/install-python/)
 
### Model Installation
Download the model from: [DeepVot Model] (https://drive.google.com/uc?export=download&id=0B69m3kcUfbmPOUE0VkpzOVF2TzQ). Then, move the model file to: `run_all/lua_scripts/model/` inside the project directory.

## Usage
For measurement just type from the run\_all folder: 
```bash
python predict.py "input wav file" "output text grid file" "start time to search" "end time to search"
```

## Example
You can try our tool using the example file in the data folder and compare it to the manual annotation.
cd into the run\_all directory and type:
```bash
python predict.py test_data/orig/bun.wav test_data/prediction/bun.TextGrid 0.0 0.2

or

python predict.py test_data/orig/bag.wav test_data/prediction/bag.TextGrid 0.56 0.65
```
