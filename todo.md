## a retrieval-only mode (free, and it targets your biggest problem)

## Deployed a RAG assistant with an automated eval harness on AWS (Bedrock, SageMaker Pipelines, Terraform)

## Investigate the current issue with retrival

    - Broken by chunking - we have a lot of not found quote in the retrived chinked passage prbably does to cut off at the middle or somtthing(done)
    - Ranked too low : try more lower ranks(done, solved with dedup + a better embedding model + expandid context)
    - Not found at all: - explore solutions like hybrid search, context heading, o rtry different embedding models.
    ## Today
    - The multihop siuuatin, where w can find one part of tehr reference and the other does not xist

, o rte otehr exist but ranked too low.. resllt, i want them both to rank higher.

dedicated a sppecal evaluation for the multi-hop question category. (22 questions currrently)
