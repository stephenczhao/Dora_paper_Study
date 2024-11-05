<h1 align="center">
    <p> DoRA: Weight-Decomposed Low-Rank Adaptation <br> [ICML2024 (Oral)]</p>
</h1>


[[`Paper`](https://arxiv.org/abs/2402.09353)] [[`Website`](https://nbasyl.github.io/DoRA-project-page/)]  [[`BibTeX`](#citation)]

# Introduction: 

Parameter efficient fine-tuning have been widely used in recent years, making training large models accessible and less computationally expensive. Specifically, [[LoRA](https://arxiv.org/pdf/2106.09685)]  - Low Rank Adaptation, have gain increasing popularity. 

**Problem**: There still often exists a gap in learning effectiveness between LoRA and full fine-tuning (FT) of models. 

**DoRA** decomposes the pre-trained weight into two components, *magnitude* and *direction*, for fine-tuning, specifically employing LoRA for directional updates to minimize the number of trainable parameters efficiently. By employing DoRA, we enhance both the learning capacity and training stability of LoRA while avoiding any additional inference overhead. DoRA consistently outperforms LoRA on fine-tuning LLaMA, LLaVA, and VL-BART on various downstream tasks, such as commonsense reasoning, visual instruction tuning, and image/video-text understanding.


# Context: LoRA and is limitations


<h1 align="center"> 
    <img src="./imgs/dora.png" width="600">
</h1>


## DoRA vs LoRA on the commonsense reasoning tasks 
| Model                 | r |    BoolQ  |  PIQA  |  SIQA  |  HellaS  |  WinoG  |  ARC-e  |  ARC-c  |  OBQA  |  Average  |
|-----------------------|-------|---------|--------|--------|-------------|--------------|---------|---------|--------|-----------|
| LLaMA-7B-LoRA		  |   32  |    67.5  |  80.8  |  78.2  |  83.4  |  80.4   |  78.0   |  62.6   |  79.1  |  76.3     |
| LLaMA-7B-DoRA	  |  [16](https://huggingface.co/sliuau/DoRA-weights/tree/main/llama_dora_commonsense_checkpoints/LLama-7B/dora_r16)   |    70.0 | 82.6 | 79.7 | 83.2 | 80.6 | 80.6 | 65.4 | 77.6 | **77.5**   |
| LLaMA-7B-DoRA 	  |  [32](https://huggingface.co/sliuau/DoRA-weights/tree/main/llama_dora_commonsense_checkpoints/LLama-7B/dora_r32)   |   69.7 | 83.4 | 78.6 | 87.2 | 81.0 | 81.9 | 66.2 | 79.2 | **78.4**   |
| LLaMA2-7B-LoRA		  |   32  |   69.8 | 79.9| 79.5| 83.6| 82.6| 79.8|64.7| 81.0| 77.6    |
| LLaMA2-7B-DoRA		  |  [16](https://huggingface.co/sliuau/DoRA-weights/tree/main/llama_dora_commonsense_checkpoints/LLama2-7B/dora_r16)   |   72.0 |83.1 |79.9| 89.1 |83.0| 84.5| 71.0 |81.2 |**80.5**  |
| LLaMA2-7B-DoRA 	  |  [32](https://huggingface.co/sliuau/DoRA-weights/tree/main/llama_dora_commonsense_checkpoints/LLama2-7B/dora_r32)   |  71.8 |83.7 |76.0 |89.1 |82.6 |83.7 |68.2| 82.4 |**79.7**   |
| LLaMA3-8B-LoRA		  |   32  |   70.8 |85.2| 79.9| 91.7 |84.3 |84.2| 71.2| 79.0| 80.8    |
| LLaMA3-8B-DoRA		  |  [16](https://huggingface.co/sliuau/DoRA-weights/tree/main/llama_dora_commonsense_checkpoints/LLama3-8B/dora_r16)   |  74.5 |88.8 |80.3| 95.5| 84.7| 90.1| 79.1| 87.2| **85.0**   |
| LLaMA3-8B-DoRA 	  |  [32](https://huggingface.co/sliuau/DoRA-weights/tree/main/llama_dora_commonsense_checkpoints/LLama3-8B/dora_r32)   |   74.6| 89.3| 79.9 |95.5| 85.6| 90.5| 80.4 |85.8 |**85.2**  |








# Useful Links
The Official PyTorch implementation of [**DoRA: Weight-Decomposed Low-Rank Adaptation**](https://arxiv.org/abs/2402.09353) 
