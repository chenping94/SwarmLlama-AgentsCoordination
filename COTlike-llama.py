import streamlit as st
from dotenv import load_dotenv
import os, re, traceback, json, time, requests
from openai import OpenAI
from pymongo import MongoClient
from swarm import Swarm, Agent

from reasoning_agents import thinker_agent


# Load environment variables
load_dotenv()

# Get configuration from .env file
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
AGENT_A_MODEL = os.getenv('AGENT_A_MODEL', 'qwen2.5:coder-7b')

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",        
    api_key="ollama"            
)

temperature=0.2
max_tokens=4096

def get_mongo_client():
    client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
    return client

def get_database(client, db_name):
    return client[db_name]

def check_for_follow_up(raw_content, step_data):
    if "Please let me know" in raw_content:
        return "Continue, Consider ALL" + important_message
    elif isinstance(step_data, dict) and step_data.get('next_action') == 'continue':
        return 'continue' + important_message
    return None

def extract_json_objects(text):
    # This regex pattern matches JSON-like structures without using recursive patterns
    json_pattern = re.compile(r'\{(?:[^{}]|\{[^{}]*\})*\}')
    return json_pattern.findall(text)

def clean_json_string(json_string):
    # Remove any text before the first '{'
    json_string = re.sub(r'^[^{]*', '', json_string)    
    # Remove any text after the last '}'
    json_string = re.sub(r'[^}]*$', '', json_string)
    # Remove any trailing commas before closing braces or brackets
    json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
    return json_string

def parse_json_safely(json_string):
    cleaned_json = clean_json_string(json_string)
    json_objects = extract_json_objects(cleaned_json)
    
    if json_objects:
        try:
            # Try to parse the last JSON object found
            return json.loads(json_objects[-1])
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {str(e)}")
    
    st.error("No valid JSON object found in the response")
    # return None  # hide if cannot find
    st.text("Raw response:")
    st.code(json_string)
    return {"title": "Error", "content": "Failed to parse response", "next_action": "final_answer"}

def run(
    starting_agent, context_variables=None, stream=False, debug=False,
    temperature=temperature,
    max_tokens=max_tokens,
    messages = None
) -> None:
    client = Swarm(client=ollama_client)
    st.text("Starting Ollama Swarm CLI 🐝")
    if messages is None:
        messages = []
    steps = []
    agent = starting_agent
    auto_continue = False
    input_counter = 0  # Counter for unique input keys

    if messages: 
        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )
        messages_response = response.messages if hasattr(response, 'messages') else response.get_messages()                
        pretty_print_messages(messages_response)
        
        st.write(response)
        
        agent = response.agent
        for message in response.messages:
            if message["role"] == "assistant":
                try:
                    content = json.loads(message["content"])
                except json.JSONDecodeError as e:
                    st.error(f"JSON decode error: {e}")
                    st.write(f"Raw content: {message['content']}")
                    return None
                st.markdown(""" this is content (json loads message["content"])""")
                st.write(content)
                return content
    
def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response =  run(thinker_agent, messages=messages, temperature=temperature, max_tokens=max_tokens)            

            raw_content = json.dumps(response)
            parsed_data = parse_json_safely(raw_content)
            return parsed_data, raw_content
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.text("Traceback:")
            st.code(traceback.format_exc())
        
        if attempt == 2:
            error_message = f"Failed to generate {'final answer' if is_final_answer else 'step'} after 3 attempts."
            return {"title": "Error", "content": error_message, "next_action": "final_answer"}
        time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt):
    client = get_mongo_client()
    db = get_database(client, "COTlike-llama")
    collection = db["steps"]
    messages = [
        # {"role": "system", "content": SYSTEM_PROMPT + important_message},
        # {"role": "user", "content": "Here is my first query: " + prompt },
        {"role": "system", "content": "You are professional."},
        {"role": "user", "content": SYSTEM_PROMPT + important_message + "Here is my first query: " + prompt },
        {"role": "assistant", "content": "Understood. I will now think step by step following the instructions, starting with decomposing the problem. I will provide my response in a single, well-formatted JSON object for each step."}
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0
    final_answer_detected = False

    while True:
        start_time = time.time()
        step_data, raw_content = make_api_call(messages, 500)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time, raw_content))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        # Store each step in MongoDB
        # collection.insert_one(step_data)
        # collection.insert_one({"steps": steps})

        # Check if a follow-up is needed
        follow_up = check_for_follow_up(raw_content, step_data)
        if follow_up:
            messages.append({"role": "user", "content": follow_up})
            if follow_up.startswith("continue"):
                step_count += 1
            continue  # Skip to the next iteration without incrementing step_count

        if step_data['next_action'] == 'final_answer':
            final_answer_detected = True
            break

        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above. Remember to respond with a single, well-formatted JSON object."})

    start_time = time.time()
    final_data, raw_content = make_api_call(messages, 300, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data['content'], thinking_time, raw_content))

    # Store the final answer in MongoDB
    # collection.insert_one(final_data)
    collection.insert_one({"steps": steps})

    yield steps, total_thinking_time

    if final_answer_detected:
        # Transfer conversation to agentA
        agentA = Agent(
            name="Evaluation Agent",
            instructions="You are an expert evaluator. Your task is to evaluate the step-by-step reasoning response towards the questions and provide an evaluation rating system from 0 to 1.",
            model=AGENT_A_MODEL
        )
        client = Swarm(client=ollama_client)
        response = client.run(
            agent=agentA,
            messages=messages
        )
        steps.append(("Evaluation Response", response.messages[-1]["content"], 0, response.messages[-1]["content"]))
        yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="COTlike-llama", page_icon="🧠", layout="wide")

    st.title("Chain-of-thoughts using llama3.2")

    st.markdown("""
    Forked from [repository](https://github.com/chenping94/COTlike-llama)     
    *Bing search "COTlike-llama" 
    """)

    st.markdown(f"**Current Configuration:**")
    st.markdown(f"- Ollama URL: `{OLLAMA_URL}`")
    st.markdown(f"- Ollama Model: `{OLLAMA_MODEL}`")

    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")

    if user_query:
        st.write("Generating response...")

        # Create empty elements to hold the generated text and total time
        response_container = st.empty()
        time_container = st.empty()

        # Generate and display the response
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time, raw_content) in enumerate(steps):
                    if title.startswith("Final Answer") or title.startswith("Evaluation Response"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                            st.markdown("**Raw Output:**")
                            st.code(raw_content, language="json")

                            # Check if a follow-up was sent
                            # follow_up = check_for_follow_up(raw_content, json.loads(raw_content))
                            parsed_data = parse_json_safely(raw_content)
                            follow_up = check_for_follow_up(raw_content, parsed_data)
                            if follow_up:
                                if follow_up.startswith("continue"):
                                    st.markdown(f"*Automatic 'continue' prompt sent*")
                                else:
                                    st.markdown(f"*Follow-up prompt sent: '{follow_up}'*")

                    st.markdown(f"*Thinking time: {thinking_time:.2f} seconds*")

            # Only show total time when it's available at the end
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")


SYSTEM_PROMPT = """You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

1. Provide a clear, concise title describing the current reasoning phase.
2. Elaborate on your thought process in the content section.
3. Decide whether to continue reasoning or provide a final answer.

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Key Instructions:
- Employ at least 5 distinct reasoning steps such as Edge Case Consideration, Precision Consideration, Alternative Hypothesis or Approach Evaluation and Elimination, etc.
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- When re-examining, employ a fundamentally different approach or perspective.
- Utilize at least 4 diverse methods to derive or verify your answer.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Quantify certainty levels for each step and the final conclusion when applicable.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses.

"""
important_message=""" 
IMPORTANT: Respond STRICTLY with a single, well-formatted JSON object for each step. Do not include any text outside the JSON object. Think STEP by STEP. 

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Example of a valid JSON response:
{"title": "Initial Problem Analysis", "content": "To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.", "next_action": "continue"}

"""

if __name__ == "__main__":
    main()
