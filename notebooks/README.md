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


### Tokenization

Tokenization is the process of splitting a text into smaller units called tokens. These tokens are the fundamental building blocks an LLM works with. 

Input Sentence: “Every moment is a beginning”

Tokens: [“Every”, “moment”, “is”, “a”, “beginning”] 

This shows how a tokenizer can split a sentence into tokens. After tokenization, each unique token is assigned a unique numerical ID. 

Here’s a simple visual showing tokenization:

<p align="center">
  <a href="https://awesomeneuron.substack.com/">
    <img src="../assets/llms_tokenization.gif" >
  </a>
</p>

### Token Embeddings

Now we have a list of numbers, but these numbers alone don’t carry any meaning. The ID “15745” for “Every” doesn’t tell the machine that “Every” is a determiner used to describe a noun. This is where embeddings help.

Token Embeddings are essentially numerical representations (vectors) of tokens basically a long list of numbers (a vector) that describes its characteristics.

<p align="center">
  <a href="https://awesomeneuron.substack.com/">
    <img src="../assets/llms_token_embedding.gif" >
  </a>
</p>

### Positional Embeddings

Imagine the sentences:

- The dog jumps on the cat.
- The cat jumps on the dog.

The words are the same, but the meaning is entirely different because their positions are different. Our numerical token IDs and token embeddings, by themselves, don’t tell the LLM anything about the order of words.

This is solved with Positional Embeddings.

Positional embeddings are another list of numbers (a vector) added to the token embeddings. These numbers are carefully designed to tell the LLM about the absolute or relative position of each token in the input sequence.

<p align="center">
  <a href="https://awesomeneuron.substack.com/">
    <img src="../assets/llms_position_embedding.gif" >
  </a>
</p>

## Code

Dive into the hands-on examples for each LLM component using interactive Jupyter notebooks.

| Topic                  | Code |
|------------------------|------|
| Tokenization           | [01_tokenization.ipynb](./notebooks/01_tokenization.ipynb) |
| Token Embeddings       | [02_token_embeddings.ipynb](./notebooks/02_token_embeddings.ipynb) |
| Positional Embeddings  | [03_positional_embeddings.ipynb](./notebooks/03_positional_embeddings.ipynb) |

## Read the full breakdown

- [A Visual Guide to LLMs (Part 1): Text to Numbers: Tokenization and Embeddings](https://awesomeneuron.substack.com/p/a-visual-guide-to-llms-part-1)
- A Visual Guide to LLMs (Part 2): Inside the Transformer Architecture