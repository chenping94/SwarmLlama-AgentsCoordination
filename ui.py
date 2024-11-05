from dotenv import load_dotenv
from openai import OpenAI
from swarm import Swarm
import json, os, requests, re, streamlit as st

from reasoning_agents import thinker_agent

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",        
    api_key="ollama"            
)

temperature=0.2
max_tokens=4096

def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "response" in chunk:
            return chunk["response"]


def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        # print(f"\033[94m{message['sender']}\033[0m:", end=" ")
        st.markdown(f"**{message['sender']}**: {message['content']}", unsafe_allow_html=True)


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
            st.markdown(f"<span style='color: purple;'>{name}({arg_str[1:-1]})</span>", unsafe_allow_html=True)
      

def run(
    starting_agent, context_variables=None, stream=False, debug=False,
    temperature=temperature,
    max_tokens=max_tokens
) -> None:
    client = Swarm(client=ollama_client)
    st.text("Starting Ollama Swarm CLI 🐝")
    response = None  # Initialize response
    
    debug_mode = st.checkbox("Debugging mode")
    
    messages = []
    messages_response = []
    steps = []
    agent = starting_agent
    auto_continue = False
    input_counter = 0  # Counter for unique input keys
    response_container = st.empty()

    while True:
        if auto_continue:
            user_input = "continue"
            auto_continue = False
        else:
            # Using a unique key for each text input
            user_input = st.text_input(f"User Input {input_counter}", "") 
            input_counter += 1  # Increment the counter for the next input
            if st.button("Launch", key=f"submit_button{input_counter}"):
                if user_input:  # Check if input is not empty
                    messages.append({"role": "user", "content": user_input})
                    # user_input = ""
                else:
                    st.warning("Please enter a query.")
                    continue
        if messages:
            response = client.run(
                agent=agent,
                messages=messages,
                context_variables=context_variables or {},
                stream=stream,
                debug=debug,
            )
        # if response is not None:  
        #     st.write("Response:", response)  # Debugging line

        if stream:
            process_and_print_streaming_response(response)
        else:
            try:
                # Access the messages safely
                messages_response = response.messages if hasattr(response, 'messages') else response.get_messages()
                if messages_response:
                    pretty_print_messages(messages_response)
                    messages.extend(messages_response)

                for message in messages_response:
                    if "###-END-###" in message.get("content", ""):
                        st.success("End of session detected. Stopping all activities.")
                        return 
                    if message["role"] == "assistant":
                        # Ensure content is present before parsing
                        if message["content"]:
                            cleaned_content = message["content"].strip()
                            """ remove non-JSON """
                            # Remove any non-JSON parts (like tool calls)
                            cleaned_content = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned_content, flags=re.DOTALL)                            
                            # Split the cleaned content into separate JSON parts
                            # json_parts = cleaned_content.split("}{")
                            # json_parts = [part.strip() for part in json_parts]
                            # Match valid JSON parts
                            json_parts = re.findall(r'{[^}]*}', cleaned_content)
                            
                            for part in json_parts:
                                # Ensure proper JSON formatting
                                if not part.startswith('{'):
                                    part = '{' + part
                                if not part.endswith('}'):
                                    part = part + '}'
                                    
                            try:
                                # content = json.loads(cleaned_content)  # Ensure content is JSON
                                content = json.loads(part)  # Ensure content is JSON
                            except json.JSONDecodeError:
                                if debug_mode:
                                    st.error(f"Failed to parse JSON: {cleaned_content}")
                                continue
                            
                            if debug_mode:
                                st.markdown(f"**{content['title']}**")
                                st.markdown(content['content'])
                            # Append step for display
                            if content not in steps:
                                steps.append({"title": content["title"], "content": content["content"]})
                                update_response_container(response_container, steps)  # Update the response container
                            
                            # Extract the JSON structure from the response
                            # cleaned_json = clean_json_string(message["content"])
                            # parsed_content = parse_json_safely(cleaned_json)
                            
                            # steps.append({"title": parsed_content.get("title"), "content": parsed_content.get("content")})
                            # st.markdown(f"**{parsed_content['title']}**")
                            # st.markdown(parsed_content['content'])


                            if "next_action" in content and content["next_action"] in ["continue", "final_answer", "transfer_to_rater_agent", "transfer_to_concluder_agent"]:
                                auto_continue = True  
                            elif "next_action" in content:
                                st.markdown(f"Next action is {content['next_action']}.")
                        else:
                            if debug_mode:
                                st.error("Received an empty response from the assistant.")

            except json.JSONDecodeError:
                if debug_mode:
                    st.error("Failed to parse response: Invalid JSON format.")
            except Exception as e:
                if debug_mode:
                    st.error(f"Error processing response: {e}")

        messages.extend(messages_response)  # Use the appropriate response messages structure
        agent = response.agent  # Update to the new agent if available
        
        # for step in steps:
        #     st.markdown(f"### {step['title']}")
        #     st.markdown(step['content'])
            
def update_response_container(container, steps):
    """Update the Streamlit response container with current steps."""
    container.empty()  # Clear the container first
    for step in steps:
        container.markdown(f"### {step['title']}")
        container.markdown(step['content'])

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

if __name__ == "__main__":
    st.title("Multi-Agent Interaction")
    run(thinker_agent)