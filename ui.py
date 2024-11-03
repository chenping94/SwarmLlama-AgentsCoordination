from dotenv import load_dotenv
from openai import OpenAI
from swarm import Swarm
import json, os, requests, streamlit as st

from reasoning_agents import planner_agent

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

        st.markdown(f"**{message['sender']}:** {message['content']}", unsafe_allow_html=True)


def run(
    starting_agent, context_variables=None, stream=False, debug=False,
    temperature=temperature,
    max_tokens=max_tokens
) -> None:
    client = Swarm(client=ollama_client)
    st.text("Starting Ollama Swarm CLI 🐝")

    messages = []
    agent = starting_agent
    auto_continue = False
    input_counter = 0  # Counter for unique input keys

    while True:
        if auto_continue:
            user_input = "continue"
            auto_continue = False
        else:
            # Using a unique key for each text input
            user_input = st.text_input(f"User Input {input_counter}", "") 
            input_counter += 1  # Increment the counter for the next input
            if st.button("Submit", key="submit_button{input_counter}"):
                if user_input:  # Check if input is not empty
                    messages.append({"role": "user", "content": user_input})
                else:
                    st.warning("Please enter a query.")

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        st.write("Response:", response)  # Debugging line

        if stream:
            process_and_print_streaming_response(response)
        else:
            try:
                # Access the messages safely
                messages_response = response.messages if hasattr(response, 'messages') else response.get_messages()
                
                pretty_print_messages(messages_response)

                for message in messages_response:
                    if message["role"] == "assistant":
                        # Ensure content is present before parsing
                        if message["content"]:
                            content = json.loads(message["content"])  # Ensure content is JSON
                            st.markdown(f"**{content['title']}**")
                            st.markdown(content['content'])

                            if "next_action" in content and content["next_action"] in ["continue", "final_answer", "transfer_to_rater_agent", "transfer_to_concluder_agent"]:
                                auto_continue = True  
                            elif "next_action" in content:
                                st.markdown(f"Next action is {content['next_action']}.")
                        else:
                            st.error("Received an empty response from the assistant.")

            except json.JSONDecodeError:
                st.error("Failed to parse response: Invalid JSON format.")
            except Exception as e:
                st.error(f"Error processing response: {e}")

        messages.extend(messages_response)  # Use the appropriate response messages structure
        agent = response.agent  # Update to the new agent if available


if __name__ == "__main__":
    st.title("Multi-Agent Interaction")
    run(planner_agent)