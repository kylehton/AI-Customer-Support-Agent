# Three-Agent Support System

A sophisticated AI-powered customer support system built with FastAPI that uses three specialized AI agents to process and respond to customer queries. The system implements a Retrieval-Augmented Generation (RAG) architecture with MongoDB for knowledge base storage.

## System Architecture

### Agents

1. **Agent 1: Triage Specialist (Query Router)**
   - Analyzes vague customer queries
   - Reformulates them into precise, technical questions
   - Optimizes queries for knowledge base searching
   - Example: "My _______ isn't working" → "Troubleshooting steps for _______ issues, problems, and basic functionality failures"

2. **Agent 2: Technical Expert (Drafting Agent)**
   - Performs semantic search on the MongoDB knowledge base
   - Retrieves relevant documentation sections
   - Synthesizes technical information into draft solutions
   - Returns structured solutions with source citations

3. **Agent 3: Communication Specialist (Critique & Refine Agent)**
   - Transforms technical drafts into customer-friendly responses
   - Adds empathy and conversational tone
   - Ensures accuracy while improving readability
   - Provides final polished customer response

## Features

- **FastAPI REST API** for lightweight server API endpoints
- **MongoDB integration** for scalable knowledge base storage
- **Semantic search** using Sentence Transformers embeddings
- **OpenAI integration** for support responses

## Requirements

- Python 3.11+
- MongoDB 7.0+
- OpenAI API Key + Credits

## Installation

### Option 2: Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup for MongoDB:**
```bash
Create Database + Collection within a Cluster on Mongo Atlas
```

3. **Configure Atlas Search**
```bash
Create a search index called "vector_index", with the custom JSON:
'
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      }
    }
  }
}
'
```

4. **Set environment variables:**
```bash
Follow .env.example to set up the necessary .env variables
```

5. **Run the application:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```



## API Usage

### Health Check
```bash
GET /health
```

### Submit Support Query
```bash
POST /support-query
Content-Type: application/json

{
  "query": "Support query here . . ."
}
```

### Response Format
```json
{
  "final_answer": "Support query answer in text here . . .",
  "sources": [
    "Source Content 1...",
    "Source Content 2..."
  ]
}
```

### Manual Testing with curl
```bash
curl -X POST "http://localhost:8000/support-query" \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I How do I clone my repository?"}'
```


### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URL` | None | MongoDB connection string |
| `DATABASE_NAME` | None | MongoDB database name |
| `COLLECTION_NAME` | None | MongoDB collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Open-source sentence transformer model |
| `OPENAI_MODEL` | gpt-3.5-turbo | OpenAI Chat Completions Model (optional) |
| `OPENAI_API_KEY` | None | OpenAI API key |
| `TOP_K` | `3` | Maximum sources returned per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum similarity for search results |
| `AGENT_TEMPERATURE` | `0.3` | Diversity/randomness of generated text |

