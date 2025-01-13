# Customer Feedback AI Agent
# Required imports
import os
from typing import List, Tuple, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

from langgraph.graph import Graph, StateGraph, MessageGraph, START, END
#from langgraph.prebuilt import ToolMessage
from langchain_core.messages import ToolMessage, FunctionMessage, AIMessage, HumanMessage
from IPython.display import Image, display
import operator
import os

from dotenv import load_dotenv
# load_dotenv() will be used on local dev envs
load_dotenv()
# os.environ Will be used in docekr. Make sure to start the docker with the -e option and provide the OPENAI_API_KEY as a environemnt variable.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "value does not exist")
# Define the states and tools our agent can use
class Task(BaseModel):
    task_type: str = ""
    name: str = ""
    description: str = ""
    time_to_complete: int = 1440 # Represetnted in minutes. Default to 1 day.

class AgentState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    current_feedback: str = ""
    sentiment: str = ""
    category: str = ""
    priority: str = ""
    next_steps: List[Task] = Field(default_factory=list)

class FeedbackCategory(str, Enum):
    PRODUCT = "product"
    SERVICE = "service"
    BILLING = "billing"
    TECHNICAL = "technical"
    OTHER = "other"

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskType(str, Enum):
    HUMAN = "human"
    AGENT = "agent"
    AUTO = "auto"

# Tool definitions
def analyze_sentiment(feedback: str) -> str:
    """
    Analyze the sentiment of customer feedback
    """
    # In a real implementation, this would use a sentiment analysis model
    # For demonstration, we'll use a simple keyword-based approach
    positive_words = ["great", "excellent", "good", "love", "perfect", "thanks"]
    negative_words = ["bad", "poor", "terrible", "hate", "awful", "worst"]

    feedback_lower =  feedback.lower() if type(feedback) is str else feedback.current_feedback.lower()
    positive_count = sum(1 for word in positive_words if word in feedback_lower)
    negative_count = sum(1 for word in negative_words if word in feedback_lower)

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    return "neutral"

def categorize_feedback(feedback: str) -> FeedbackCategory:
    """
    Categorize the type of feedback
    """
    #feedback_lower = feedback.lower()
    feedback_lower =  feedback.lower() if type(feedback) is str else feedback.current_feedback.lower()

    if any(word in feedback_lower for word in ["product", "feature", "quality"]):
        return FeedbackCategory.PRODUCT
    elif any(word in feedback_lower for word in ["service", "support", "staff"]):
        return FeedbackCategory.SERVICE
    elif any(word in feedback_lower for word in ["bill", "price", "cost"]):
        return FeedbackCategory.BILLING
    elif any(word in feedback_lower for word in ["bug", "error", "technical"]):
        return FeedbackCategory.TECHNICAL
    return FeedbackCategory.OTHER

def determine_priority(sentiment: str, category: str) -> Priority:
    """
    Determine the priority level of the feedback
    """

    if sentiment == "negative" and category in [FeedbackCategory.TECHNICAL, FeedbackCategory.BILLING]:
        return Priority.HIGH
    elif sentiment == "negative":
        return Priority.MEDIUM
    return Priority.LOW

def generate_next_steps(feedback: str, sentiment: str, category: str, priority: str) -> List[str]:
    """
    Generate recommended next steps based on the analysis
    """
    steps = []

    if priority == Priority.HIGH:
        steps.append(Task(task_type=TaskType.HUMAN, name="Escalate", description="Escalate to relevant department immediately", time_to_complete=1440))

    if sentiment == "negative":
        steps.append(Task(task_type=TaskType.HUMAN, name="Refer Concern", description="Prepare customer response addressing concerns", time_to_complete=1440))
    elif sentiment == "positive" or sentiment == "neutral":
        steps.append(Task(task_type=TaskType.AGENT, name=" Generate Content", description="Write a personolized article", time_to_complete=3))

    if category == FeedbackCategory.TECHNICAL:
        steps.append(Task(task_type=TaskType.AUTO, name="Open Support Ticket", description="Create technical support ticket", time_to_complete=1))
        steps.append(Task(task_type=TaskType.AGENT, name="Generate Content", description="RAG based solution advisor", time_to_complete=3))

    if not steps:
        steps.append(Task(task_type=TaskType.AUTO, name="Log feedback", description="File feedback in customer database for monthly review", time_to_complete=1))

    return steps

# Define the agent's workflow
def process_feedback(state: AgentState) -> AgentState:
    """
    Process a piece of customer feedback through the entire workflow
    """
    feedback = state.current_feedback
    # Analyze sentiment
    sentiment = analyze_sentiment(feedback)
    state.sentiment = sentiment

    # Categorize feedback
    category = categorize_feedback(feedback)
    state.category = category

    # Determine priority
    priority = determine_priority(sentiment, category)
    state.priority = priority

    # Generate next steps
    next_steps = generate_next_steps(feedback, sentiment, category, priority)
    state.next_steps = next_steps
    return state

# Create the graph
def create_feedback_agent(state: AgentState) -> StateGraph:
    #workflow = Graph()
    workflow = StateGraph(AgentState)

    # Add the main feedback processing node
    workflow.add_node("process_feedback", process_feedback)

    # Define the edges
    workflow.add_edge(START, "process_feedback")
    workflow.add_edge("process_feedback", END)

    return workflow

# A more complex agent with multiple node to apply the logic:
# will need to adapt some of the impemetation - mostly all function should get and return AgentState...
def create_complex_feedback_agent() -> Graph:
    workflow = Graph()

    # Add multiple processing nodes
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("categorize", categorize_feedback)
    workflow.add_node("prioritize", determine_priority)
    workflow.add_node("generate_actions", generate_next_steps)

    # Define the flow between nodes
    # The output of each node becomes the input for the next
    workflow.add_edge(START, "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "categorize")
    workflow.add_edge("categorize", "prioritize")
    workflow.add_edge("prioritize", "generate_actions")
    workflow.add_edge("generate_actions", END)

    return workflow

# Example usage
def run_feedback_analysis(feedback: str) -> Dict:
    """
    Run the feedback analysis workflow for a given piece of feedback
    """
    # Initialize the state
    initial_state = AgentState(current_feedback=feedback)

    # Create and run the workflow
    agent = create_feedback_agent(initial_state)
    compiled_agent = agent.compile()
    try:
      display(Image(compiled_agent.get_graph().draw_mermaid_png()))
    except Exception:
      print("Cannot draw graph. This requires some extra dependencies")
    #agent = create_complex_feedback_agent()
    final_state = compiled_agent.invoke(initial_state)
    # Return the results
    return {
        "feedback": feedback,
        "sentiment": final_state['sentiment'],
        "category": final_state['category'],
        "priority": final_state['priority'],
        "next_steps": final_state['next_steps']
    }

# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_feedbacks = [
        "The product quality is excellent and the features are just what I needed!",
        "I've been having technical issues for days and nobody has helped me yet.",
        "The billing department charged me twice and I'm still waiting for a refund.",
    ]

    print("Running Customer Feedback Analysis:")
    print("-" * 50)

    for feedback in test_feedbacks:
        results = run_feedback_analysis(feedback)
        print(f"\nFeedback: {feedback}")
        print(f"Sentiment: {results['sentiment']}")
        print(f"Category: {results['category']}")
        print(f"Priority: {results['priority']}")
        print("Next Steps:")
        for step in results['next_steps']:
            print(f"- {step}")
        print("-" * 50)