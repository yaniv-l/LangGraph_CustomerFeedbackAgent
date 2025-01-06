# Customer Feedback Agent

A smart agent system designed to process, analyze, and respond to customer feedback efficiently.

## Overview

The Customer Feedback Agent is an automated solution that helps businesses manage and derive insights from customer feedback. It processes feedback from multiple channels and provides intelligent responses and analytics.
Based on the LangGraph framework, from the LangGraph framework, this agent is designed to process, analyze, and respond to customer feedback efficiently.
This Agent simulate agents working with large languageis, but works all localy to speed up processing for demo and avoid cost and charges.

## Features

- Automated feedback analysis processing: 
    - Sentiment analysis
    - Topic extraction
    - Category classification
    - Next best action recommendation
- LangGraph based
- All local execution - does not require LLM access to speed up execution for demo purposes
- Microservice architecture
- RESTful API

## Getting Started

### Prerequisites

- Python 3.11
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CustomerFeedbackAgent.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. create a hapi.txt file with your api keys (you can have muliple keys seperated by new lines). 
  - Note - This API keys are to access the Aget API server and are not used for the LLM.
2. If want to used LLM:
  2.1. Create a .env file, or set an environment variable, with your LLM api key.
  2.2. Adjust the agent implemetation to use the LLM

## Usage

Start the API server (consider expose via encrypted tunnel):
```bash
python api.py
```

To run the agent directly via python:
```bash
python feedback_agent.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue in the GitHub repository.

## Acknowledgments

- Thanks to all contributors
- Inspired by the LangGraph framework
- Built with modern AI and NLP technologies