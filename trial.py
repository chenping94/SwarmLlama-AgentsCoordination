import streamlit as st
import requests
import json
import os
from swarm import Swarm
from openai import OpenAI  # Ensure you import this
from reasoning_agents import planner_agent


OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')

ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def make_api_call(messages):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2}
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {str(e)}")
        return None

def generate_response(user_query):
    client = Swarm(client=ollama_client)
    
    messages = [{"role": "user", "content": user_query}]
    
    # Process response through planner agent
    response = client.run(agent=planner_agent, messages=messages)
    
    return response

def main():
    st.title("Multi-Agent Interaction")
    
    user_query = st.text_input("Enter your query:")
    
    if st.button("Submit"):
        if user_query:
            with st.spinner("Processing..."):
                response = generate_response(user_query)
                if response:
                    for message in response.get('messages', []):
                        if message["role"] == "assistant":
                            try:
                                content = json.loads(message["content"])
                                st.markdown(f"**{content['title']}**")
                                st.markdown(content['content'])
                            except json.JSONDecodeError:
                                st.error("Failed to parse response.")
                else:
                    st.error("No response received.")
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()
