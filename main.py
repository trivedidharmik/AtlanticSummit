import pandas as pd
from crewai import Task, Crew
from agents.agent_factory import create_agent
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load data
agent_df = pd.read_excel("data/Agent_Profiles.xlsx")
articles_df = pd.read_excel("data/Data Set.xlsx")

# Load prompt template
with open("prompts/prompt_template.txt") as f:
    prompt_template = f.read()

responses = []

for _, agent_row in agent_df.iterrows():
    agent_name = agent_row.get("Name") or agent_row.index[0]
    agent_profile = agent_row.dropna().to_string(index=False)
    agent = create_agent(agent_name, agent_profile)

    for _, article_row in articles_df.iterrows():
        title = article_row.get("Title", "")
        summary = article_row.get("Summary", "")
        
        prompt = prompt_template.format(
            agent_name=agent_name,
            agent_profile=agent_profile,
            article_title=title,
            article_summary=summary
        )

        task = Task(description=prompt, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()

        responses.append({
            "agent": agent_name,
            "article": title,
            "response": result
        })

# Save responses
df = pd.DataFrame(responses)
df.to_csv("responses/agent_article_responses.csv", index=False)
print("All responses saved to 'responses/agent_article_responses.csv'")