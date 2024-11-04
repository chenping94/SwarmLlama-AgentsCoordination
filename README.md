# SwarmLlama-AgentsCoordination
Experimental, Forked from https://github.com/coleam00/ai-agents-masterclass  

Prerequisites:
You must have your own model! 
The model used here are qwen2.5-coder:7b and qwen2.5:3b 

Steps to run:

First, create your own RSS feed databases:
1. python load_sql_data.py 

2. python run.py

Have Fun!! 


p/s：Performance wise really not good.. tool calls transfer to other agents sometimes wont work.
Honestly, giving decision when to use tool calls or transfer agent to the LLM itself is Very unreliable 0-0 
