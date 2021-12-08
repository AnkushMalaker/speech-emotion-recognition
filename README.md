# Implementation of Speech emotion recognition with deep convolutional neural networks  

<h1 align="center">
  <br>
Speech emotion recognition with deep convolutional neural networks
  <br>
</h1>

> **Speech emotion recognition with deep convolutional neural networks**<br>
> Dias Issa, M. Fatih Demirci, Adnan Yazici<br>
>
> **Abstract:** *The speech emotion recognition (or, classification) is one of the most challenging topics in data science. Inthis work, we introduce a new architecture, which extracts mel-frequency cepstral coefficients, chroma-gram, mel-scale spectrogram, Tonnetz representation, and spectral contrast features from sound files anduses them as inputs for the one-dimensional Convolutional Neural Network for the identification of emo-tions using samples from the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS),Berlin (EMO-DB), and Interactive Emotional Dyadic Motion Capture (IEMOCAP) datasets. We utilize anincremental method for modifying our initial model in order to improve classification accuracy. All ofthe proposed models work directly with raw sound data without the need for conversion to visual repre-sentations, unlike some previous approaches. Based on experimental results, our best-performing modeloutperforms existing frameworks for RAVDESS and IEMOCAP, thus setting the new state-of-the-art. Forthe EMO-DB dataset, it outperforms all previous works except one but compares favorably with that onein terms of generality, simplicity, and applicability. Specifically, the proposed framework obtains 71.61%for RAVDESS with 8 classes, 86.1% for EMO-DB with 535 samples in 7 classes, 95.71% for EMO-DB with 520samples in 7 classes, and 64.3% for IEMOCAP with 4 classes in speaker-independent audio classificationtasks.*

<h4 align="center"><a href="https://www.sciencedirect.com/science/article/abs/pii/S1746809420300501">link to paper</a></h4>


## Training your model
### Dataset: 
- Audio files can be put in the `./train_data` folder

### Parameters
- Pending

### Example Usage
``` python trian.py -B 64 15```  
(Runs the training function with default parameters, batch size of 64 for 15 epochs)

## Saved Model
- The model will be saved post training in the "saved_model" directory. 
- Checkpoints feature pending

## Environment Setup
Tested with:  
- Python=3.9  
- Tensorflow=2.7  
- Librosa=0.81
- scikit-learn=1.0.1

## Model Description

## To-do:  

## Credits and acknowledgements:


## Citation

>
    Dias Issa, M. Fatih Demirci, Adnan Yazici,
    Speech emotion recognition with deep convolutional neural networks,
    Biomedical Signal Processing and Control,
    Volume 59,
    2020,
    101894,
    ISSN 1746-8094,
    https://doi.org/10.1016/j.bspc.2020.101894.
    (https://www.sciencedirect.com/science/article/pii/S1746809420300501)
    Abstract: The speech emotion recognition (or, classification) is one of the most challenging topics in data science. In this work, we introduce a new architecture, which extracts mel-frequency cepstral coefficients, chromagram, mel-scale spectrogram, Tonnetz representation, and spectral contrast features from sound files and uses them as inputs for the one-dimensional Convolutional Neural Network for the identification of emotions using samples from the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), Berlin (EMO-DB), and Interactive Emotional Dyadic Motion Capture (IEMOCAP) datasets. We utilize an incremental method for modifying our initial model in order to improve classification accuracy. All of the proposed models work directly with raw sound data without the need for conversion to visual representations, unlike some previous approaches. Based on experimental results, our best-performing model outperforms existing frameworks for RAVDESS and IEMOCAP, thus setting the new state-of-the-art. For the EMO-DB dataset, it outperforms all previous works except one but compares favorably with that one in terms of generality, simplicity, and applicability. Specifically, the proposed framework obtains 71.61% for RAVDESS with 8 classes, 86.1% for EMO-DB with 535 samples in 7 classes, 95.71% for EMO-DB with 520 samples in 7 classes, and 64.3% for IEMOCAP with 4 classes in speaker-independent audio classification tasks.
    Keywords: Speech emotion recognition; Deep learning; Signal processing

