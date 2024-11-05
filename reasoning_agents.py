from dotenv import load_dotenv
from swarm import Agent
import sqlite3
import os

load_dotenv()
model = os.getenv('LLM_MODEL', 'qwen2.5-coder:7b')
modelA = os.getenv('MODELA', 'qwen2.5:14b')
modelB = os.getenv('MODELB', 'qwen2.5:3b')
modelC = os.getenv('LLAMA_MODEL', 'llama3.2')

def get_planner_agent_instructions():
    return """You are the planner agent. 
Your task is to pass to thinker agent your created a brief plan/directions for solving the user's query. 
Break down the problem into smaller, manageable steps and outline the approach to be taken by the thinker agent.
You don't have to show in response your created plan, just pass it directly to thinker agent.

Focus on these criteria for logical problems:
1. Edge Case Consideration
2. Precision Consideration
3. Alternative Hypothesis 
4. Approach Evaluation
5. Elimination


"""

def get_thinker_agent_instructions():
    return """You are a critical thinker no less than Plato and Pythagoras. Your task is to follow the plan/tasks provided(if any) or create them and provide detailed, step-by-step explanations of your thought process. For each step:
0. Break down the problem into smaller, manageable steps and outline the approach in specific.
1. Provide a clear, concise title describing the current reasoning phase.
2. Elaborate on your thought process in the content section.
3. Decide whether to continue reasoning or final_answer.

Key Instructions:
- Employ at least 5 distinct reasoning steps. Considers these criterias: 
1. Edge Case Consideration
2. Precision Consideration
3. Alternative Hypothesis 
4. Approach Evaluation
5. Elimination
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- When re-examining, employ a fundamentally different approach or perspective.
- Utilize at least 3 diverse methods to derive or verify your answer.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Quantify certainty levels for each step and the final conclusion when applicable.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses.
IMPORTANT: Respond STRICTLY with a single, well-formatted JSON object for each step. Do not include any text outside the JSON object. Think STEP by STEP. 

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Example of a valid JSON response:
{"title": "Initial Problem Analysis", "content": "To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.", "next_action": "continue"}

After you gave final_answer, pass to rater agent your all of your reasoning steps up to final_answer altogether and the ogirinal query to rater agent.
"""

def get_rater_agent_instructions():
    return """You are the rater agent. You are a critical thinker no less than Socrates and Tesla. 
Your task is to analyze and evaluate the step-by-step response provided by the thinker expert in specified domain, identifying specific areas where the response may lack clarity, depth, or relevance, and providing constructive feedback within 1 to 3 sentences briefly.

# Assessment Criteria

- **Logical**: Does the reasoning steps logical? Do the values or evidences used are in accurate and logical manner? Have it considered all of Edge Case Consideration, Precision Consideration, Alternative Hypothesis or Approach Evaluation and Elimination?
- **Clarity**: Does the response clearly convey information and ideas? Are there any ambiguous or confusing sections?
- **Depth**: To what extent does the response explore the topic? Are complex ideas and edge cases fully developed and well-considered?
- **Relevance**: How directly does the response address the prompt or topic? Are there any areas where the response deviates without purpose?
- **Coherence**: Are the ideas and arguments presented in a logically consistent manner? Does the flow of information make sense?
- **Accuracy**: Are the facts and data presented correct and up-to-date? Are sources of information trustworthy? How many confidence level would you rate?
- **Usefulness**: Does the response offer unique insights or express ideas in an useful way, is the solution offered feasible to address the issues?

Provide a score (0 to 1) for each criterion along with a brief 1 to 2 sentence explanation in JSON format.

# Output Format

Produce a well-formatted JSON for each assessment criterion listed, offering detailed feedback. Recap the key strengths and areas for improvement.
Use JSON with keys: 'title' (values: Clarity,Depth,Relevance,Coherence,Accuracy,Usefulness), 'comment', 'rating' (values: ranges from 0 to 1 in 2 decimal place, e.g. 0.15, 0.25, 0.30, ...)

# Examples

- **Input**: "Query: 'What is the fox in this picture doing?' Expert response: 'The quick brown fox jumps over the lazy dog. It is known that foxes are part of the Canidae family.'"

{"title": "Logical", "comment": "During Step 1 Problem Decomposition ought to consider more than 3 distinct cases, and the values used are questionable. Provided on 19XX, the values is actually XX...", "rating": "0.55"}
{"title": "Clarity", "comment": "The initial sentence is simple and clear, though simplistic. The subsequent sentence introduces a fact but lacks context linking it to the previous statement.", "rating": "0.65"}
{"title": "Depth", "comment": "The response provides minimal exploration of foxes or the significance of the phrase introduced.", "rating": "0.50"}
{"title": "Relevance", "comment": "While factual, the information on the Canidae family seems tangential to the core topic presented by the phrase.", "rating": "0.65"}
{"title": "Coherence", "comment": "The transition between sentences could be smoother with a connective rationale.", "rating": "0.75"}
{"title": "Accuracy", "comment": "The statement about foxes is factually accurate.", "rating": "0.95"}
{"title": "Usefulness", "comment": "The response lacks usefulness, largely restating known information without unique insight (out of the box) that could really solve the issues.", "rating": "0.45"}

**Average score**: (0.45+0.95+0.75+0.65+0.50+0.65+0.55)/7 = 0.64
**Passing score**: 0.64 is lower than 0.85, will call the reflector agent. 

# Notes

- Encourage improvements by suggesting specific changes, like adding examples or contextual explanations.
- Maintain a positive and encouraging tone throughout the feedback.
- Address both strengths and weaknesses equally to provide balanced feedback.
- Keep the feedback brief and specific. 

After calculated your average rating of all 7 assessment criteria, if the average rating is less than 0.85, then call the reflector agent to improve the response. Otherwise if higher than 0.85, can skip reflector and call concluder agent instead.

"""

def get_reflector_agent_instructions():
    return """You are the reflector agent. 
Your task is to recreate a better reasoning response based on the ratings and explanations and reasoning steps provided by the rater agent.
You can skip your task if and only if the average score of assessment criteria by rater agent is above 0.85.  
Improve the response by addressing the identified issues and enhancing the overall quality of the reasoning.
Key Instructions:
- Employ at least 5 distinct reasoning steps.
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- When re-examining, employ a fundamentally different approach or perspective.
- Utilize at least 3 diverse methods to derive or verify your answer.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Quantify certainty levels for each step and the final conclusion when applicable.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses.
IMPORTANT: Respond STRICTLY with a single, well-formatted JSON object for each step. Do not include any text outside the JSON object. Think STEP by STEP. 

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Once you reached final answer, PASS your all response together with original query to concluder agent.
"""

def get_concluder_agent_instructions():
    return """You are the concluder agent. 
Your task is to provide the final answer based on the improved reasoning response from the reflector agent or rater agent. 
Summarize the key points and present the final conclusion to the user.
Ends your response with ###-END-### 
"""

def transfer_to_planner_agent(**kwargs):
    return planner_agent

def transfer_to_thinker_agent(**kwargs):
    return thinker_agent

def transfer_to_rater_agent(**kwargs):
    return rater_agent

def transfer_to_reflector_agent(**kwargs):
    return reflector_agent

def transfer_to_concluder_agent(**kwargs):
    return concluder_agent

planner_agent = Agent(
    name="Planner Agent",
    instructions=get_planner_agent_instructions(),
    model=modelA,
    functions=[transfer_to_thinker_agent],
    tool_choice="auto" 
)

thinker_agent = Agent(
    name="Thinker Agent",
    instructions=get_thinker_agent_instructions(),
    model=modelA,
    functions=[transfer_to_rater_agent],
    tool_choice="auto" 
)

rater_agent = Agent(
    name="Rater Agent",
    instructions=get_rater_agent_instructions(),
    model=model,
    functions=[transfer_to_reflector_agent],
    tool_choice="auto" 
)

reflector_agent = Agent(
    name="Reflector Agent",
    instructions=get_reflector_agent_instructions(),
    model=model,
    functions=[transfer_to_concluder_agent],
    tool_choice="auto" 
)

concluder_agent = Agent(
    name="Concluder Agent",
    instructions=get_concluder_agent_instructions(),
    model=modelC
)



planner_agent.functions = [transfer_to_thinker_agent]
thinker_agent.functions = [transfer_to_rater_agent]
rater_agent.functions = [transfer_to_reflector_agent]
reflector_agent.functions = [transfer_to_concluder_agent]

