import io
import os

import arxiv
from diskcache import Cache
import requests
from autogen.agentchat.contrib.img_utils import _to_pil, get_image_data
from openai import OpenAI
from pydantic import BaseModel
from PyPDF2 import PdfReader
from typing_extensions import Annotated

# class BlogSectionModel(BaseModel):
#     sectionTitle: Annotated[str, "Title of this section of the blog"]
#     sectionContent: Annotated[str, "Content of this section of the blog"]


# class BlogModel(BaseModel):
#     title: Annotated[str, "Title of the blog"]
#     imageUrl: Annotated[str, "Url for image of the blog"]
#     sections: Annotated[
#         list[BlogSectionModel], f"List of blog sections with title and content"
#     ]


# def save_blog_to_db(
#     host: str, database: str, user: str, password: str, blog: BlogModel
# ):
#     conn = psycopg2.connect(
#         host=host,
#         database=database,
#         user=user,
#         password=password,
#     )
#     sql = """INSERT INTO blogs(blog_json)
#             VALUES(%s) RETURNING vendor_id;"""

#     vendor_id = None

#     try:
#         with psycopg2.connect(
#             host=host,
#             database=database,
#             user=user,
#             password=password,
#         ) as conn:
#             with conn.cursor() as cur:
#                 # execute the INSERT statement
#                 cur.execute(sql, (blog.json(),))

#                 # get the generated id back
#                 rows = cur.fetchone()
#                 if rows:
#                     vendor_id = rows[0]

#                 # commit the changes to the database
#                 conn.commit()
#     except (Exception, psycopg2.DatabaseError) as error:
#         print(error)
#     finally:
#         return vendor_id


def save_blog_to_file(
    arxiv_id,
    blogJson: str,
) -> str:
    path = f"./_output/output-{arxiv_id}"
    file = f"blog-{arxiv_id}.json"
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            with open(os.path.join(path, file), "w") as f:
                f.write(blogJson)
        else:
            with open(os.path.join(path, file), "a") as f:
                f.write(blogJson)
        return "Succesfully saved blog json to file"
    except Exception as error:
        return "Failed to save blog json to file"


def generate_image(
    client: OpenAI, model: str, prompt: str, size: str, quality: str, n: int
) -> str:
    try:
        cache = Cache(".cache/")  # Create a cache directory
        key = (model, prompt, size, quality, n)
        if key in cache:
            image_url = cache[key]
        else:
            # If not in cache, compute and store the result
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
            )
            image_url = response.data[0].url
            cache[key] = image_url
        return image_url
    except Exception as error:
        return error.__str__


def create_image_and_save(
    arxiv_id: str,
    image_file_name: str,
    client: OpenAI,
    model: str,
    prompt: str,
    size: str,
    quality: str,
    n: int,
) -> str:
    """
    Generate an image using OpenAI's DALL-E model and cache the result.

    This function takes a prompt and other parameters to generate an image using OpenAI's DALL-E model.
    It checks if the result is already cached; if so, it returns the cached image data. Otherwise,
    it calls the DALL-E API to generate the image, stores the result in the cache, and then returns it.

    Args:
        client (OpenAI): The OpenAI client instance for making API calls.
        model (str): The specific DALL-E model to use for image generation.
        prompt (str): The text prompt based on which the image is generated.
        size (str): The size specification of the image. TODO: This should allow specifying landscape, square, or portrait modes.
        quality (str): The quality setting for the image generation.
        n (int): The number of images to generate.

    Returns:
    str: The image data as a string, either retrieved from the cache or newly generated.

    Note:
    - The cache is stored in a directory named '.cache/'.
    - The function uses a tuple of (model, prompt, size, quality, n) as the key for caching.
    - The image data is obtained by making a secondary request to the URL provided by the DALL-E API response.
    """
    # Function implementation...
    try:
        img_url = generate_image(
            client=client, model=model, prompt=prompt, size=size, quality=quality, n=n
        )
        img_data = get_image_data(img_url)
        img = _to_pil(img_data)
        file = f"{image_file_name}"
        path = f"./output-{arxiv_id}"
        if not os.path.exists(path):
            os.makedirs(path)
        img.save(os.path.join(path, file))

        return img_url
    except Exception as error:
        return error.__str__


def save_md_file(
    arxiv_id,
    content,
) -> str:
    path = f"./_output/output-{arxiv_id}"
    file = f"blog-{arxiv_id}.md"
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            with open(os.path.join(path, file), "w") as f:
                f.write(content)
        else:
            with open(os.path.join(path, file), "a") as f:
                f.write(content)
        return "Succesfully saved markdown file"
    except:
        return "Failed to save markdown file"


class ArxivDocument(BaseModel):
    title: Annotated[str, "Title of the article"]
    content: Annotated[str, "Contect of the article"]


def fetch_arxiv(
    url: Annotated[str, "arXiv url on the form https://arxiv.org/abs/0000.00000"]
) -> ArxivDocument:
    client = arxiv.Client()
    id = url.split("/")[-1]
    search = arxiv.Search(id_list=[id])
    results = next(client.results(search=search))

    response = requests.get(results.pdf_url)

    # Create a BytesIO object from the content
    pdf_content = io.BytesIO(response.content)

    # Use PyPDF2 to read the PDF content
    pdf_reader = PdfReader(pdf_content)
    content = ""

    # Iterate over each page and extract text
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        content += "\n\n " + page.extract_text()

    return ArxivDocument(title=results.title, content=content)
