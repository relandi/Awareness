# Awareness

The official implementation of the Awareness model, as in the paper [CognitiveNet: Enriching Foundation Models with Emotions and Awareness](https://doi.org/10.1007/978-3-031-35681-0_7).

## Installing the conda environment

```sh
conda create -y -n awareness python=3.8 pip
conda activate awareness

pip install -e ./awareness/
pip install git+https://github.com/openai/CLIP.git
pip install -r ./requirements.txt
```
