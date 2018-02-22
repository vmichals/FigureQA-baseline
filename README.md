# FigureQA-Baseline
This repository contains the TensorFlow implementations of the CNN-LSTM, Relation Network and text-only baselines for our paper _[FigureQA: An Annotated Figure Dataset for Visual Reasoning](https://arxiv.org/abs/1710.07300)_ \[[Project Page](http://datasets.maluuba.com/FigureQA)\].

If you use code from this repository for your scientific work, please cite
```
Kahou, S. E., Michalski, V., Atkinson, A., Kadar, A., Trischler, A., & Bengio, Y. (2017). Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300.
```
If you use the Relation Network implementation, please also cite
```
Santoro, A., Raposo, D., Barrett, D. G., Malinowski, M., Pascanu, R., Battaglia, P., & Lillicrap, T. (2017). A simple neural network module for relational reasoning. In Advances in neural information processing systems (pp. 4974-4983).
```
## Getting started
The setup was tested with python 3, tensorflow 1.4 and 1.6.0-rc1. We recommend using the [Anaconda Python Distribution](https://anaconda.org/anaconda/python).
1. Create a virtual machine, e.g. via
```
conda create -p ~/venvs/figureqa python=3
```
2. Activate the environment:
```
source activate ~/venvs/figureqa
```
3. Install dependencies:
```
conda install numpy tqdm six matplotlib pandas
pip install tensorflow-gpu 
```

4. [Download](http://datasets.maluuba.com/FigureQA/dl) the FigureQA data set _tar.gz_ archives (unextracted) into a directory named _FigureQA_.

5. Clone the baseline repository somewhere locally (here we're using $HOME/workspace)
```
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:vmichals/FigureQA-Baseline.git
```

## Training and Evaluation
### Training a model
Run the training script for the model. It takes the following required arguments:
 * --data-path: the directory, in which you placed the tar.gz archives of FigureQA, referred to as  _DATAPATH_ in the following.
 * --tmp-path: a temporary directory, in which the script will extract the data (preferably on fast storage, such as an SSD or a RAM disk), from now on referred to as TMPPATH
 * --model: the model you want to train (_rn_, _cnn_ or _text_), from now on referred to as MODEL
 * --num-gpus: the number of GPUs to use (in the same machine), from now on referred to as NUMGPU
 * --val-set: the validation set to use for early-stopping (validation1 or validation2), from now on referred to as VALSET
 * (additional configuration options can be found in the \*.json files in the _cfg_ subfolder)
```
cd ~/workspace/FigureQA-baseline
python -u train_model_on_figureqa.py --data-path DATAPATH --tmp-path TMPPATH \
    --model MODEL --num-gpus NUMGPU --val-set VALSET
```
### Resuming interrupted training
To resume interrupted training, run the resume script, which takes the following required arguments:
 * --data-path: same as for the training script
 * --tmp-path: same as for the training script
 * --train-dir: the training directory created by the training script (a subfolder of the train\_dir), from now on referred to as _TRAINDIR_
 * --num-gpus: same as for the training script
 * --resume-step: the time-step from which to resume (check training directory for the model-TIMESTEP.meta file with the largest TIMESTEP), from now on referred to as RESUMESTEP
```
cd ~/workspace/FigureQA-baseline
python -u resume_train_model_on_figureqa.py --train-dir TRAINDIR --resume-step RESUMESTEP \
    --num-gpus NUMGPU --data-path DATAPATH --tmp-path TMPPATH
```
### Testing
To evaluate a trained model, run the eval script, which takes the following required arguments:
 * --train-dir: same as for the resume script
 * --meta-file: the meta file of the trained model, e.g. "model_val_best.meta" from the 
 * --data-path: same as for the resume and training script
 * --tmp-path: same as for the resume and training script

 Example: 
 ```
 cd ~/workspace/FigureQA-baseline
 python -u ./eval_model_on_figureqa.py  --train-dir TRAINDIR --meta-file METAFILE \
     --partition test2 --data-path DATAPATH --tmp-path TMPPATH
 ```
For the test1 and test2 partitions, the script will dump your predictions to a csv file. 
To get the test accuracy, please submit the file [here](mailto:figureqa@microsoft.com?subject=Evaluate%20FigureQA%20test%20results) and we will get back to you with the results as soon as possible.
