from crewai import Agent

def create_agent(agent_name, background, llm=None):
    return Agent(
        role=agent_name,
        goal="Respond to health-related articles based on your personal profile.",
        backstory=background,
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )