This is the repo accompanying my ECS7013P Deep Learning for Audio and Music coursework assignment submission: Automatic Drum Transcription with Data Augmentation.

You can clone this repo and its submodules using the following command

```
git clone --recurse-submodules -j8 https://github.com/pelinski/TransformerADT.git
```

In order to run the code, you can use the provided environment:

```
conda env create -f environment.yml
```

and activate it using

```
conda activate groove
```

For some reason, this usually gives errors, so alternatively you can create your own environment with python 3.6

```
conda create --name groove python=3.6
source activate groove
```

and install the following packages:

```
pip install visual_midi
pip install tables
pip install magenta==1.1.7 --use-deprecated=legacy-resolver
pip install note_seq

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
pip3 install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
conda install -c conda-forge fluidsynth
pip install pyFluidSynth
pip install wandb
conda install -c anaconda wget
pip install bokeh
pip install pandas
pip install PySoundFile
pip install colorcet
pip install holoviews
pip install ipyparams
```

You will also need the datasets and evaluators, that you can get from here:

```
wget link
wget link
unzip datasets.zip
unzip evaluator.zip
```
