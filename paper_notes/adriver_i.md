# [ADriver-I: A General World Model for Autonomous Driving](https://arxiv.org/abs/2311.13549)

_February 2024_

tl;dr: LLM-based e2e driver to predict future action and vision autoregressively.

#### Overall impression
The paper proposed an interesting problem of jointly rolling out action and vision, based on past action-vision pairs. In a way it performs e2e planning by directly predicting control signals (steer angles and ego speed), like TCP and Transfuser, directly predicting planning resutls without constructing any scenre representation (at least not explicitly).

The good thing about this paper is that it jointly predicts action and video. In contrast, [GAIA-1](gaia_1.md) only generates future videos but ignores control signal prediction. [DriveDreamer](drive_dreamer.md) conditions the future video generation heavily on prior information such as structured/vectorized perception results. This is taken one step further by [Panacea](panacea.md).

The author claim that the paper creates "infinite driving". However the prediction is in a piecewise fashion, predicting action with a world model initialized by a pretrained VLM, and then using VDM (vision difusion model) with the predicted action. The two pieces are trained separately without joint finetuning.

Another less clean design of the paper is the use of the language is a bit too heavy. There are a lot of tricks to convert the control signals to text (such as converting float to integers, and controlling num of digits), especially downgrading precise numbers to quantitative descriptions. This is not aligned with the goal of precise AD.

The **biggest drawback** of this paper, as I see it, is the action conditioning of the VDM. The VDM is NOT conditioned on precise *quantitative* action parameters but rather relied on *qualitative* text description. The lack of finegrained controllability makes it less powerful to act as a neural simulator. --> This needs to be improved, and qualitative metrics should be established to ensure geometric consistencies of the actions. 

#### Key ideas
- Architecture
	- World model is a VLM Similar to Llava. The LLM component is based on Vicuna-7B (SFT'ed from Llama2) and 
	- Video diffusion model (VDM), similar to [VideoLDM](videoldm.md). It is conditioned on historical frames and control signals. 
		- The text encoder of VDM is not powerful enough to understand the control signal, so it is **transcribed to qualitative descroption by GPT3.5**. --> This is NOT reasonable, and must be improved. See the notes of [Drive into the Future](drive_wm.md) for training tricks.

#### Technical details
- Training of VLM (MLLM)
	- Pretraining: LLM is frozen, vision encoder and adapator are trained.
	- SFT: Vision encoder is frozen, LLM and vision adaptor are trained.
- Training of VDM
	- Pretraining on private dataset
	- Finetuned on nuscenes

#### Notes
- The paper also lack a lot of precise details.
	- What is the delta T between frames?
	- What is multiround conversation?
