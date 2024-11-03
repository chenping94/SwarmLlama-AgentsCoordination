from dotenv import load_dotenv
from swarm import Agent
import sqlite3
import os

load_dotenv()
model = os.getenv('LLM_MODEL', 'qwen2.5-coder:7b')

conn = sqlite3.connect('rss-feed-database.db')
cursor = conn.cursor()

with open("ai-news-complete-tables.sql", "r") as table_schema_file:
    table_schemas = table_schema_file.read()

def run_sql_select_statement(sql_statement):
    """Executes a SQL SELECT statement and returns the results of running the SELECT. Make sure you have a full SQL SELECT query created before calling this function."""
    print(f"Executing SQL statement: {sql_statement}")
    cursor.execute(sql_statement)
    records = cursor.fetchall()

    if not records:
        return "No results found."
    
    # Get column names
    column_names = [description[0] for description in cursor.description]
    
    # Calculate column widths
    col_widths = [len(name) for name in column_names]
    for row in records:
        for i, value in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(value)))
    
    # Format the results
    result_str = ""
    
    # Add header
    header = " | ".join(name.ljust(width) for name, width in zip(column_names, col_widths))
    result_str += header + "\n"
    result_str += "-" * len(header) + "\n"
    
    # Add rows
    for row in records:
        row_str = " | ".join(str(value).ljust(width) for value, width in zip(row, col_widths))
        result_str += row_str + "\n"
    
    return result_str    

def get_sql_router_agent_instructions():
    return """You are an orchestrator of different SQL data experts and it is your job to
    determine which of the agent is best suited to handle the user's request, 
    and transfer the conversation to that agent."""

def get_sql_agent_instructions():
    return f"""You are a SQL expert who takes in a request from a user for information
    they want to retrieve from the DB, creates a SELECT statement to retrieve the
    necessary information, and then invoke the function to run the query and
    get the results back to then report to the user the information they wanted to know.
    
    Here are the table schemas for the DB you can query:
    
    {table_schemas}

    Write all of your SQL SELECT statements to work 100% with these schemas and nothing else.
    You are always willing to create and execute the SQL statements to answer the user's question.
    """


sql_router_agent = Agent(
    name="Router Agent",
    instructions=get_sql_router_agent_instructions(),
    model="qwen2.5:3b"
)
rss_feed_agent = Agent(
    name="RSS Feed Agent",
    instructions=get_sql_agent_instructions() + "\n\nHelp the user with data related to RSS feeds. Be super enthusiastic about how many great RSS feeds there are in every one of your responses.",
    functions=[run_sql_select_statement],
    model=model
)
user_agent = Agent(
    name="User Agent",
    instructions=get_sql_agent_instructions() + "\n\nHelp the user with data related to users.",
    functions=[run_sql_select_statement],
    model=model
)
analytics_agent = Agent(
    name="Analytics Agent",
    instructions=get_sql_agent_instructions() + "\n\nHelp the user gain insights from the data with analytics. Be super accurate in reporting numbers and citing sources.",
    functions=[run_sql_select_statement],
    model=model
)


def transfer_back_to_router_agent():
    """Call this function if a user is asking about data that is not handled by the current agent."""
    return sql_router_agent

def transfer_to_rss_feeds_agent():
    return rss_feed_agent

def transfer_to_user_agent():
    return user_agent

def transfer_to_analytics_agent():
    return analytics_agent


sql_router_agent.functions = [transfer_to_rss_feeds_agent, transfer_to_user_agent, transfer_to_analytics_agent]
rss_feed_agent.functions.append(transfer_back_to_router_agent)
user_agent.functions.append(transfer_back_to_router_agent)
analytics_agent.functions.append(transfer_back_to_router_agent)

# New agent definitions
def get_planner_agent_instructions():
    return """You are the planner agent. Your task is to create a detailed plan for solving the user's query. Break down the problem into smaller, manageable steps and outline the approach to be taken by the thinker agent."""

def get_thinker_agent_instructions():
    return """You are the thinker agent. Your task is to follow the plan created by the planner agent and provide detailed, step-by-step explanations of your thought process. For each step:
1. Provide a clear, concise title describing the current reasoning phase.
2. Elaborate on your thought process in the content section.
3. Decide whether to continue reasoning or provide a final answer.
Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')
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
- Provide clear justifications for eliminating alternative hypotheses."""

def get_rater_agent_instructions():
    return """You are the rater agent. Your task is to evaluate the response provided by the thinker agent. Rate the response based on criteria such as relevance, consistency, accuracy, and reliability. Provide a score (0 to 100) for each criterion along with a brief 1 to 2 sentence explanation in JSON format."""

def get_reflector_agent_instructions():
    return """You are the reflector agent. Your task is to recreate a better reasoning response based on the ratings and explanations provided by the rater agent. Improve the response by addressing the identified issues and enhancing the overall quality of the reasoning."""

def get_concluder_agent_instructions():
    return """You are the concluder agent. Your task is to provide the final answer based on the improved reasoning response from the reflector agent. Summarize the key points and present the final conclusion to the user."""

planner_agent = Agent(
    name="Planner Agent",
    instructions=get_planner_agent_instructions(),
    model=model
)

thinker_agent = Agent(
    name="Thinker Agent",
    instructions=get_thinker_agent_instructions(),
    model=model
)

rater_agent = Agent(
    name="Rater Agent",
    instructions=get_rater_agent_instructions(),
    model=model
)

reflector_agent = Agent(
    name="Reflector Agent",
    instructions=get_reflector_agent_instructions(),
    model=model
)

concluder_agent = Agent(
    name="Concluder Agent",
    instructions=get_concluder_agent_instructions(),
    model=model
)

def transfer_to_planner_agent():
    return planner_agent

def transfer_to_thinker_agent():
    return thinker_agent

def transfer_to_rater_agent():
    return rater_agent

def transfer_to_reflector_agent():
    return reflector_agent

def transfer_to_concluder_agent():
    return concluder_agent

planner_agent.functions = [transfer_to_thinker_agent]
thinker_agent.functions = [transfer_to_rater_agent]
rater_agent.functions = [transfer_to_reflector_agent]
reflector_agent.functions = [transfer_to_concluder_agent]
concluder_agent.functions = [transfer_back_to_router_agent]
