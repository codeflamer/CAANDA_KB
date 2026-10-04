## a retrieval-only mode (free, and it targets your biggest problem)

## Deployed a RAG assistant with an automated eval harness on AWS (Bedrock, SageMaker Pipelines, Terraform)

## Investigate the current issue with retrival

    - Broken by chunking - we have a lot of not found quote in the retrived chinked passage prbably does to cut off at the middle or somtthing
    - Ranked too low : try more lower ranks
    - Not found at all: - explore solutions like hybrid search, context heading, o rtry different embedding models.
