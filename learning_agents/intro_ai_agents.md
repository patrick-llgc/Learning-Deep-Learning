# [Intro to AI Agent with OpenClaw](https://www.bilibili.com/video/BV1UqPQzXEmy)

By Hungyi-Lee, Professor of NTU 

# Notes
* ![](./assets/history.jpeg)
* OpenClaw (AI Agent) is NOT AI or LLM. It is the interface between human and LLM. 
* OpenClaw is the non-AI part of AI agent. It is the embodiment to interact with the real world.
	* The smartness comes from the LLM, not the agent part. 
	* OpenClaw does agent **orchestration**.
* ![](./assets/agent_and_ai.jpeg)
* The real thing that LLM does is 尻取り (Shiritori) or 词语接龙. Even after SFT or RL. 
	* LLM is a wise person trapped in a cell, with a slip passing in from a window.
* Context window is limited. The longer the context is, even if it is not yet hitting the upper limit, the performance would degrade. 
* System prompt: always in the prompt, before user prompt.
	* SOUL.md
	* IDENTITY.md
	* USER.md
	* MEMORY.md: long term memory
	* what tools can be used (and how to use them)
	* AGENTS.md (the model's behavior)
	* What skills can be used
	* where to find past memories
* ![](./assets/system_prompt.jpeg)
* System level information can be modified by human, but better to be modified by agents for consistency.
* Every single round of conversation, it contanins history since the beginning of the session of all past rounds of conversation.
	* Most LLM APIs are stateless.
	* very common way to optimize is context summarization: summary + recent context. Lose exact detail but keeps semantic states.
	* Prompt Caching (prefix caching) can also save compute via kv caching of exactly the same prefix. Modern LLM. This is typically supported by modern LLM APIs.
* ![](./assets/multiround_conversation.jpeg)
* Anterograde amnesia
	* LLM has anterograde amnesia (short term memory loss)
	* 50 first dates, and memento
* How does AI agent use computer
	* tool calling by LLM 
	* coding (script creating) by LLM
	* Claw only executes the tool calling or scripts.
	* LLM is good at giving instructions (via Shell cmd), and Claw is good at executing them. 
* Securty and defense of dangerous behavior
	* can be prompted from MEMORY.md, but not always realiable.
	* double check at execution, on OpenClaw level
* ![](./assets/security.jpeg)
* Skill
	* 	SOP of a workflow
*  A good Skill defines:
	*  what problem it solves
	*  when to use it
	*  inputs
	*  outputs
	*  steps
	*  tools
	*  constraints
	*  success criteria
*  OpenClaw would scan certain dir to find relevant skills, populate that into system prompt --> how does openClaw does the matching?
	*  apply eligibility gates → inject metadata into prompt → model picks relevant skill
	*  So openClaw CANNOT select relevant skills, and system prompt might contain completely irrelevant skills. LLM later picks the reelvant skills, based on discription.
	*  Skills are loaded as needed.
* the usage of subagents (`session_spawn`)
	* simplified context window via hierarchy
	* reduced latency
	* diff subagents can have diff LLM backend and also temperature
	* converts **deep serial chain** in to **wide shallow tree**
* When to use subagents
	* Can be parallizable
	* can be multi-staged
	* Need expertise (coding, debugging, review, etc)
	* When coordination cost > parallel benefit, do NOT use subagents
	* Converting a brain into a team.
* ![](./assets/subagent.jpeg)
* Memory
	* Long term memory in MEMORY.md
	* Short term memory in memory/yyyy-mm-ddy.md

* Memory behavior (see [AGENTS.md](https://github.com/openclaw/openclaw/blob/main/docs/reference/templates/AGENTS.md))
	* today + yesterday for recent memory
	* MEMORY.md for long term memory
	* essentially we are doing RAG with memory md files.

```
# Memory (from AGENTS.md)
You wake up fresh each session. These files are your continuity:

	* Daily notes: memory/YYYY-MM-DD.md (create memory/ if needed) — raw logs of what happened
	* Long-term: MEMORY.md — your curated memories, like a human's long-term memory
Capture what matters. Decisions, context, things to remember. Skip the secrets unless asked to keep them.
```
* ![](./assets/memory.jpeg)
* Heartbeat mechanism
	* Ping OpenClaw to send LLM preset command at fixed intervals (if set to be so in HEARTBEAT.md)
* Cron job get ML system to wait
	* LLM calls cron jobs
	* Ping OpenClaw 
* Context compaction
	* Context sent to another LLM to compress
	* Pruning techniques
		* soft trim (keep beginning and end part)
		* hard clear (redact tool output entirely)
* Best practice: important stuff to write to system prompt, not only in context. Context compression will sometimes lose the info.

# Thinking tidbits
- Prompt caching is essentially prefill kv caching.
	- By caching frequently used instructions, context, or long documents, it reduces latency by up to 85% and cuts input token costs by up to 90% for subsequent requests.
	- Drastically improves Time to First Token (TTFT) by skipping processing of the cached prefix.
	- This happens especially in the same session, and very much like the streaming style VLM.
	- Best Practices: Place **static content** (system prompts, context, long documents) at the beginning of the prompt and user-specific inputs at the end to maximize cache hits.
- Claude named their lightweight, balanced and most powerful models, Haiku, Sonnet and Opus, to mimic the length of these poets. [link](https://platform.claude.com/docs/en/about-claude/models/overview)