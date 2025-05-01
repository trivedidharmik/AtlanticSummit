import os
import asyncio
import re
import pandas as pd
from dotenv import load_dotenv
from crewai import Task, Crew, LLM
from agents.agent_factory import create_agent

# Load environment variables
load_dotenv()

import litellm
#litellm._turn_on_debug()

# Setup LLM
ollama_llm = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434",
    api_key="",
    provider="ollama"
)

# Load data
agent_df = pd.read_excel("data/Agent_Profiles.xlsx")
articles_df = pd.read_excel("data/Data Set.xlsx")

# Load prompt template
with open("prompts/prompt_template.txt") as f:
    prompt_template = f.read()

# Prepare async task execution
async def run_task(agent_name, agent_profile, article_title, article_summary):
    from crewai import Task, Crew

    agent = create_agent(agent_name, agent_profile, llm=ollama_llm)

    prompt = prompt_template.format(
        agent_name=agent_name,
        agent_profile=agent_profile,
        article_title=article_title,
        article_summary=article_summary
    )

    task = Task(
        description=prompt,
        agent=agent,
        expected_output="Answer the 3 questions clearly and concisely."
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    try:
        result = str(await crew.kickoff_async())

        # Extract the structured answers using regex
        likelihood_match = re.search(r'1\..*?(\d)', result)
        confidence_match = re.search(r'2\..*?(\d)', result)
        emotion_match = re.search(r'3\..*?[:\-]?\s*(.*)', result)

        likelihood = int(likelihood_match.group(1)) if likelihood_match else None
        confidence = int(confidence_match.group(1)) if confidence_match else None
        emotion = emotion_match.group(1).strip() if emotion_match else None

        return {
            "agent": agent_name,
            "article": article_title,
            "likelihood": likelihood,
            "confidence": confidence,
            "emotion": emotion
        }

    except Exception as e:
        print(f"Task failed for {agent_name} on '{article_title}':", e)
        return None

async def main():
    tasks = []
    for _, agent_row in agent_df.iterrows():
        agent_name = agent_row.get("Agent") or agent_row.index[0]
        agent_profile = agent_row.dropna().to_string(index=False)
        for _, article_row in articles_df.iterrows():
            title = article_row.get("Title", "")
            summary = article_row.get("Summary", "")
            tasks.append(run_task(agent_name, agent_profile, title, summary))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for result in results:
        if isinstance(result, dict):
            responses.append(result)
        else:
            print("Task failed:", result)

    df = pd.DataFrame(responses)
    df.to_csv("responses/agent_article_responses.csv", index=False)
    print("All responses saved to 'responses/agent_article_responses.csv'")

if __name__ == "__main__":
    asyncio.run(main())