# [LLM, Vision Tokenizer and Vision Intelligence, by Lu Jiang](https://mp.weixin.qq.com/s/Hamz5XMT1tSZHKdPaCBTKg)

_March 2024_

tl;dr: Summarize the the main idea of the paper with one sentence.

### LLM and Diffusion Models in Video Generation
* Two primary technological approaches in video gen: diffusion-based techniques and those based on language models. 
	* Very recently, VideoPoet uses language model, while WALT uses a diffusion. Both uses the [MagVit V2](magvit_v2.md) tokenizer.
* Diffusion iterations: pixel diffusion --> latent diffusion --> latent diffusion with a transformer backbone to replace UNet backbone(DiT).
* Diffusion now dominates ~90% of research, due to the open-sourced stable diffusion.
* Language modeling of video actually predate diffusion, with early instances like ImageGPT and subsequent developments like DALL-E, although DALL-E2 transitioned to diffusion. 
* While diffusion and large language models (LLMs) are categorized separately for ease of understanding, the boundaries between them are increasingly blurred. Diffusion technologies are progressively incorporating techniques from language models, making the distinction less clear.


### LLMs and True Visual Intelligence
- A more refined **visual tokenizer** integration with LLMs can be a method to achieve visual intelligence. The text modality (language) already includes a tokenizer, "natural language" system. A language system is needed for the visual domain. 
- Although language models have potential, they don't understand the specific goals of the generation tasks. The presence of a tokenizer establishes connections between tokens to clarify the task at hand for the model, enhancing the LLM's ability to utilize its full potential. Thus, **if a model doesn't understand its current generation task, the issue lies not with the language model itself but in our failure to find a method for it to comprehend the task.** --> Why?
- [MagVit V2](magvit_v2.md) shows that a good tokenizer connected to a language model can immediately achieve better results than the best diffusion models.
- To enhance control, such as precise control and generation through conversation, and even achieve the mentioned intelligence, we need to train a **best foundation model** as possible. **Precise control is typically a downstream issue.** The better the foundation model, the better the downstream tasks will be. Different foundation models have unique features that may eliminate certain problems in new foundation models.
- **Stable diffusion has not yet successfully scaled**. Transformers scale up more easily and have many existing learning recipes. The largest diffusion models have ~7-8B parameters, but the largest transformer models are 2 orders of magnitude larger with ~trillion params.


### References
* MaskGIT enhances image generation efficiency through parallel decoding, significantly accelerating the process while improving image quality.
* Unlike diffusion or auto-regressive models, Muse operates in a discrete token space, employing masked training that achieved SOTA perf and efficiency at the time.
* Magvit (Masked Generative Video Transformer): Introduced a 3D tokenizer in research, quantifying videos into spatio-temporal visual tokens.
