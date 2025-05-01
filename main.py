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

# Function to extract clean answers using multiple strategies
def extract_structured_answers(result):
    # Store result in a file for debugging if needed
    with open("responses/last_raw_response.txt", "w") as f:
        f.write(result)
    
    # Strategy 1: Look for answers with specific format patterns
    likelihood = None
    confidence = None
    emotion = None
    
    # Try to find simple numbered answers first (1. 5)
    simple_numbered = re.findall(r'(\d+)[\.:\)]\s*(\d+)', result)
    if len(simple_numbered) >= 2:
        try:
            likelihood = int(simple_numbered[0][1])
            confidence = int(simple_numbered[1][1])
        except (ValueError, IndexError):
            pass
    
    # If not found, try more complex patterns
    if likelihood is None:
        likelihood_match = re.search(r'(?:1\.|\blikelihood\b|take the COVID-19 vaccine).*?(\d)[^\d]', result, re.IGNORECASE)
        if likelihood_match:
            try:
                likelihood = int(likelihood_match.group(1))
            except ValueError:
                likelihood = None
    
    if confidence is None:
        confidence_match = re.search(r'(?:2\.|\bconfidence\b).*?(\d)[^\d]', result, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = int(confidence_match.group(1))
            except ValueError:
                confidence = None
    
    # Multiple strategies for emotion extraction
    # Strategy 1: Look for structured response after "3."
    emotion_match = re.search(r'(?:3\.|\bemotion\b|feel emotionally)[^:]*[:]\s*(.*?)(?:\n\d|\Z)', result, re.IGNORECASE | re.DOTALL)
    if emotion_match:
        emotion = emotion_match.group(1).strip()
    
    # Strategy 2: Find sentences containing emotional words
    if not emotion or len(emotion) < 10:
        emotional_words = r'(?:feel|emotion|emotionally|makes me|article is|article makes)'
        emotion_fallback = re.search(f'{emotional_words}[^\n.]*([^.]*\.)', result, re.IGNORECASE)
        if emotion_fallback:
            emotion = emotion_fallback.group(1).strip()
    
    # Strategy 3: Look for full sentences after "3."
    if not emotion or len(emotion) < 10:
        after_3 = re.search(r'3\.[^.]*?(.*?\.)', result, re.IGNORECASE | re.DOTALL)
        if after_3:
            emotion = after_3.group(1).strip()
    
    # Clean up the emotion text
    if emotion:
        # Remove any numbering or question prefixes
        emotion = re.sub(r'^[^a-zA-Z]*', '', emotion)
        # Limit to one sentence if multiple sentences exist
        sentence_match = re.search(r'([^.!?]+[.!?])', emotion)
        if sentence_match:
            emotion = sentence_match.group(1).strip()
        # Remove any trailing incomplete sentences
        emotion = re.sub(r'[^.!?]+$', '', emotion).strip()
        # Final sanity check - if too short, discard
        if len(emotion) < 10:
            emotion = None
    
    # Validate ranges for likelihood and confidence
    if likelihood is not None and (likelihood < 1 or likelihood > 5):
        likelihood = None
    if confidence is not None and (confidence < 1 or confidence > 5):
        likelihood = None
        
    return likelihood, confidence, emotion

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
        expected_output="Answer the 3 questions with clear numbers for 1 and 2, and a single sentence for 3."
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    try:
        result = str(await crew.kickoff_async())
        print(f"\nRaw result for {agent_name} on '{article_title}':")
        print("-" * 50)
        print(result)
        print("-" * 50)

        likelihood, confidence, emotion = extract_structured_answers(result)
        
        print(f"Extracted answers for {agent_name} on '{article_title}':")
        print(f"Likelihood: {likelihood}")
        print(f"Confidence: {confidence}")
        print(f"Emotion: {emotion}")
        print("-" * 50)

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