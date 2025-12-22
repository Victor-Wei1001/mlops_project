# mlops_project
02476 course
Project: English-to-Chinese Translation with MLOps 
In this project, we want to build a translator that converts English to Chinese. We will fine-tune the T5-small model using a subset of the WMT19 English-Chinese dataset. This dataset is a standard for machine translation tasks.
T5 is a versatile text-to-text model from Huggingface. While it can already handle multiple languages, we aim to specifically fine-tune it for the English-to-Chinese language pair. To achieve this, we will use the Transformers framework as our required third-party package, along with PyTorch Lightning to keep our training code organized.

Our main focus is on building a solid MLOps pipeline. To reach this goal, we will use:

Cookiecutter for a clean and standard project structure.

DVC for handling and versioning our translation data.

Docker for containerizing our code to ensure it runs everywhere.

Weights and Biases to track our experiments and monitor model performance.
