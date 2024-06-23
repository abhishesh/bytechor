# ByteChor

## Hackathon

### Install dependencies

* Ollama - Install Ollama package for Mac/Linux

Compared to mistral and llama3, llama3 is very good at answering from the context

```shell
ollama pull nomic-embed-text
ollama pull mistral
ollama pull llama3
ollama list
```

* UV package manager python
```shell
brew install uv
```

* Poppler - PDF Reading

```shell
brew install poppler
```

* Install miniconda and create python3.12 environment

```shell
conda create --name py312 python=3.12
conda activate py312
```

* python modules

```shell
uv pip install -r requirements.txt
```

### run project

```shell
python -m aichatbot --pdf_resource_path aichatbot.resources --model_name llama3 --input_question "who is president of India"
python -m aichatbot --pdf_resource_path aichatbot.resources --model_name llama3 --input_question "what is liquid clustering"
```
