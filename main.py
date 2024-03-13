from fastapi import FastAPI

from agents.blogify_autogen import BlogInput, BlogOutput, blogify_arxiv

app = FastAPI()


@app.post("/api/v1/blogify")
def blogify(input: BlogInput) -> BlogOutput:
    return blogify_arxiv(input=input)
