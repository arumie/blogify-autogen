import os
import autogen

from autogen.cache import Cache
from pydantic import BaseModel, SecretStr
from typing_extensions import Annotated
from openai import OpenAI
from tools.utils import ArxivDocument, create_image_and_save, fetch_arxiv, save_md_file


class BlogInput(BaseModel):
    openaiApiKey: Annotated[SecretStr, "OpenAI Apikey"]
    link: Annotated[str, "arXiv link on the form https://arxiv.org/abs/0000.00000"]


class BlogOutput(BaseModel):
    markdown: Annotated[str, "Resulting markdown of the blog"]


def blogify_arxiv(input: BlogInput) -> BlogOutput:
    if "OPENAI_API_KEY" in os.environ:
        apiKey = os.environ["OPENAI_API_KEY"]
    else:
        apiKey = input.openaiApiKey.get_secret_value()

    config_list = [
        {
            "model": "gpt-4o",
            "api_key": apiKey,
            "tags": ["chat", "gpt-4"],
        },
        {
            "model": "dall-e-3",
            "api_key": apiKey,
            "tags": ["images", "dalle"],
        },
    ]

    chat_config_list = autogen.filter_config(config_list, {"tags": ["chat"]})

    llm_config = {
        "timeout": 600,
        "seed": 1,
        "config_list": chat_config_list,
        "temperature": 0.9,
    }

    arxiv_blogifier_agent = autogen.AssistantAgent(
        name="Blogger",
        system_message="Fetch content from arXiv article and create blogs. Reply TERMINATE when the task is done.",
        llm_config=llm_config,
        description="Generates blogs from arXiv articles",
    )

    user_proxy = autogen.UserProxyAgent(
        name="Editor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "_output", "use_docker": False},
        llm_config=llm_config,
        system_message=""""You are an editor of a blog. 
    You ensure that the blogger generates a blog given an arXiv article and ensures that an image for the blog is also generated and saved.
    Reply TERMINATE if the task has been solved at full satisfaction.
    Otherwise, reply the reason why the task is not solved yet.""",
    )

    @user_proxy.register_for_execution()
    @arxiv_blogifier_agent.register_for_llm(
        description="Generate image and save as file from prompt. Returns image url."
    )
    def save_image_file_llm(
        arxiv_id: Annotated[str, "ID of the arXiv article"],
        prompt: Annotated[str, "Prompt used to generate image."],
    ) -> Annotated[str, "Resulting image URL"]:
        apiKey = os.environ["OPENAI_API_KEY"] if "OPENAI_API_KEY" in os.environ else input.openaiApiKey.get_secret_value()
        client = OpenAI(api_key=apiKey)
        return create_image_and_save(
            arxiv_id=arxiv_id,
            image_file_name=f"image-{arxiv_id}.jpg",
            client=client,
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="standard",
            n=1,
        )

    @user_proxy.register_for_execution()
    @arxiv_blogifier_agent.register_for_llm(description="Save markdown file")
    def save_md_file_llm(
        arxiv_id: Annotated[str, "ID of the arXiv article"],
        content: Annotated[str, "Markdown content to be saved in the file"],
    ) -> Annotated[str, "String saying if the operation was successful or not"]:
        return save_md_file(arxiv_id, content)

    @user_proxy.register_for_execution()
    @arxiv_blogifier_agent.register_for_llm(
        description="Fetch content form arxiv given url"
    )
    def fetch_arxiv_llm(
        url: Annotated[str, "arXiv url on the form https://arxiv.org/abs/0000.00000"]
    ) -> Annotated[ArxivDocument, "Document with title and content"]:
        return fetch_arxiv(url)

    with Cache.disk() as cache:
        # start the conversation
        user_proxy.initiate_chat(
            arxiv_blogifier_agent,
            message=f""""
                Create a blog from the the following article: '{input.link}'.
                Create an image for the blog using the title as a prompt and include in the blog via image url. 
                Include the original authors and link to arXiv article in the blog.
                Store the resulting blog as Markdown in a file.
            """,
            cache=cache,
        )

    # Read from file with name output-<arxiv_id>/blog-<arxiv_id>.md
    arxiv_id = input.link.split("/")[-1]
    with open(f"./_output/output-{arxiv_id}/blog-{arxiv_id}.md", "r") as file:
        content = file.read()
    return BlogOutput(markdown=content)
