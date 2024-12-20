from dotenv import load_dotenv
from openai import OpenAI
from swarm import Swarm
import json

from sql_agents import sql_router_agent
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

def run(
    starting_agent, context_variables=None, stream=False, debug=False,
    temperature=temperature,
    max_tokens=max_tokens
) -> None:
    client = Swarm(client=ollama_client)
    print("Starting Ollama Swarm CLI 🐝")

    messages = []
    agent = starting_agent
    auto_continue = False

    while True:
        if auto_continue:
            user_input = "continue"
            auto_continue = False
        else:
            user_input = input("\033[90mUser\033[0m: ")+"After your planning, pass to thinker agent. Thinker agent, after you presented all your reasoning steps, pass to rator agent."
        messages.append({"role": "user", "content": user_input})

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        if stream:
            response = process_and_print_streaming_response(response)
        else:
            pretty_print_messages(response.messages)
            
            for message in response.messages:
                if message["role"] == "assistant":
                    try:
                        # Try to parse the content as JSON
                        content = json.loads(message["content"])
                        if "next_action" in content and (content["next_action"] == "continue" or content["next_action"] =="final_answer" or content["next_action"] =="transfer_to_rater_agent"):
                            auto_continue = True  
                        elif "next_action" in content:
                            print(f"Next action is {content['next_action']}.")
                    except json.JSONDecodeError:
                        pass

        messages.extend(response.messages)
        agent = response.agent

if __name__ == "__main__":
    # run(sql_router_agent)
    run(thinker_agent)