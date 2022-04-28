# Audio-Visual Emotion Recognition

## About
Minor Project, VI Semester, 2021.

## Description

In this work, we study FER in a cross-domain few-shot learning setting, where only a few frames of novel classes from the target domain are required as a reference. 

In particular, we aim to identify unseen emotions, in a 4-way one shot learning fashion. Our work follows few-shot learning principles that enables learning of an embedding network, which later used to recognize unseen emotions. 

We make use of multi-modal encoder architecture that is capable of processing audio and video inputs. Our embedding network is trained on four emotions, namely, happy, sad, angry and surprised, and tested on four unseen emotions, namely contempt, neutral, disgust and fear. 

During training, the goal is to construct a rich feature space of emotions, which enables the embedding network to better differentiate one emotion from the other. 

At test time, it uses that knowledge to differentiate between unseen emotions and recognize them.

## Dataset

We have used MEAD dataset for training and testing purpose.

## Training the Model

```python
python train.py --data_root <path/to/MEAD/dataset> --checkpoint_dir <path/to/save/checkpoints> --model_checkpoint_path <path/to/model>
```

## Evaluating the Model

```python
python test.py --data_root <path/to/MEAD/dataset> --model_checkpoint_path <path/to/model>
```