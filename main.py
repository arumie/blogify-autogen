from fastapi import FastAPI
from fastapi_healthchecks.api.router import HealthcheckRouter, Probe

from agents.blogify_autogen import BlogInput, BlogOutput, blogify_arxiv

app = FastAPI()

app.include_router(
    HealthcheckRouter(
        Probe(
            name="readiness",
            checks=[],
        ),
        Probe(
            name="liveness",
            checks=[],
        ),
    ),
    prefix="/health",
)


@app.post("/api/v1/blogify")
def blogify(input: BlogInput) -> BlogOutput:
    return blogify_arxiv(input=input)
