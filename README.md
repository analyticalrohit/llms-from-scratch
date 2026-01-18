<p align="center">
  <a href="https://www.linkedin.com/in/analyticalrohit" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://awesomeneuron.substack.com/" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Substack-%23006f5c.svg?style=for-the-badge&logo=substack&logoColor=FF6719" alt="Substack">
  </a>
   <a href="https://x.com/_rohit_tiwari_" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white" alt="X">
  </a>
     <a href="https://www.youtube.com/@awesomeneuron?sub_confirmation=1" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white" alt="Youtube">
  </a>
     <a href="https://topmate.io/analyticalrohit" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/analyticalrohit/analyticalrohit/refs/heads/main/assets/topmate_logo.png" alt="Topmate">
  </a>
</p>

# LLMs from Scratch

## Overview
This repository is a hands on guide to building a ChatGPT like LLM in PyTorch. It breaks the architecture into simple parts and explains each one step by step.

## LLM Architecture

Let us have a birds eye view of the Generative Pretrained Transformer (GPT) like LLM architecture.

Example: *Every moment is a beginning*

<p align="center">
  <a href="https://awesomeneuron.substack.com/">
    <img src="./assets/llms_birds_eye_view.gif" >
  </a>
</p>


LLMs work by predicting one word or token at a time. LLMs generate text iteratively. Each predicted word token is appended to the previous input to form the context for the next prediction.

## Contents

- [Tokenization](./notebooks/README.md#tokenization)
- [Token Embeddings](./notebooks/README.md#token-embeddings)
- [Positional Embeddings](./notebooks/README.md#positional-embeddings)
- Self Attention Mechanism
- Multi-Head Self Attention
- FeedForward Neural Networks
- Residual Connections
- Layer Normalization
- Transformer Block
- Cross Entropy Loss
- Training and Generation

## Code Notebook

Dive into the hands-on examples for each LLM component using interactive Jupyter notebooks.

| Topic                     | Code |
|---------------------------|------|
| Tokenization              | [01_tokenization.ipynb](./notebooks/01_tokenization.ipynb) |
| Token Embeddings          | [02_token_embeddings.ipynb](./notebooks/02_token_embeddings.ipynb) |
| Positional Embeddings     | [03_positional_embeddings.ipynb](./notebooks/03_positional_embeddings.ipynb) |
| Self Attention Mechanism  | TODO |
| Multi-Head Self Attention | TODO |
| FeedForward Neural Networks| TODO |
| Residual Connections      | TODO |
| Layer Normalization       | TODO |
| Transformer Block         | TODO |
| Cross Entropy Loss        | TODO |
| Training and Generation   | TODO |

## Install Dependencies
```
pip install -r requirements.txt
```
If you're installing torch with CUDA support, make sure to use the correct installation command from [PyTorch's official website](https://pytorch.org/get-started/locally/), as some versions require a specific installation method.

## Blog Post

Read the full breakdown and insights in the accompanying blogs.

- [A Visual Guide to LLMs (Part 1): Text to Numbers: Tokenization and Embeddings](https://awesomeneuron.substack.com/p/a-visual-guide-to-llms-part-1)
- A Visual Guide to LLMs (Part 2): Inside the Transformer Architecture

## Newsletter
<div style="text-align: left;">
📌 Join 10,000+ ML enthusiasts and professionals from 150+ countries.<br>
✅ Learn AI for FREE with visuals, easy-to-follow insights.<br>
✅ Get cutting-edge topics like GenAI, RAGs, and LLMs in your inbox every week.
</div>
<br>
<div align="center">

[![Subscribe to AwesomeNeuron Newsletter](https://raw.githubusercontent.com/analyticalrohit/analyticalrohit/5ab83e498b11eefe57c91bc4f4cac10414276920/assets/subscribe_button.svg)](https://awesomeneuron.substack.com/)

</div>
<div style="text-align: left;">
    <a href="https://awesomeneuron.substack.com/">
        <img src="https://raw.githubusercontent.com/analyticalrohit/analyticalrohit/refs/heads/main/assets/awesomeneuron_logo.png" alt="AwesomeNeuron Newsletter">
</div>
<p align="center">
  <a href="https://awesomeneuron.substack.com/">
    <img src="https://raw.githubusercontent.com/analyticalrohit/analyticalrohit/refs/heads/main/assets/awesomeneuron_blog.gif" alt="AwesomeNeuron Newsletter">
  </a>
</p>

## Contributing

We welcome contributions! If you have improvements, new notebooks, or fixes to suggest:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/YourTopic`.
3. Add or update notebooks in the `notebooks/` folder.
4. Commit your changes: `git commit -m 'Add or update YourTopic notebook'`.
5. Push your branch: `git push origin feature/YourTopic`.
6. Open a pull request for review.

## License

This project is licensed under [MIT License](./LICENSE)

---

⭐️ If you find this repository helpful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=analyticalrohit/llms-from-scratch&type=date&legend=top-left)](https://www.star-history.com/#analyticalrohit/llms-from-scratch&type=date&legend=top-left)

Keywords: AI, Machine Learning, Deep Learning, PyTorch, Generative AI, LLMs, Transformers