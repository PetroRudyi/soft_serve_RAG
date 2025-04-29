import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import base64
import mimetypes
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config, CacheParameters

# Init VDBs
def load_vdb():
    try:
        path = Config.vector_db_path
        client = chromadb.PersistentClient(path=path)

        text_collection = client.get_collection(name="text_collection")

        image_loader = ImageLoader()
        CLIP = OpenCLIPEmbeddingFunction()
        image_collection = client.get_collection(name="image_collection",
                                                 embedding_function=CLIP,
                                                 data_loader=image_loader)
        return text_collection, image_collection
    except Exception as e:
        raise RuntimeError(f"Failed to load Vector DB: {str(e)}")


# Get images uri from VDB
def image_uris(image_collection, query_text, max_distance=None, max_results=Config.max_retrieved_images):
    results = image_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['uris', 'distances']
    )

    filtered_uris = []
    for uri, distance in zip(results['uris'][0], results['distances'][0]):
        if max_distance is None or distance <= max_distance:
            filtered_uris.append(uri[3:]) # fix path

    return filtered_uris

# Get relevant news from VDB
def text_uris(text_collection, query_text, max_distance=None, max_results=Config.max_retrieved_text_documents):
    results = text_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['documents', 'distances', 'metadatas']
    )

    filtered_texts = []
    for doc, distance, metadatas, title in zip(results['documents'][0], results['distances'][0],
                                               results['metadatas'][0], results['ids'][0]):
        if max_distance is None or distance <= max_distance:
            full_doc_text = f"Title: {title}\nURL: {metadatas.get('url', '')}\nText: {doc}"
            filtered_texts.append(full_doc_text)

    return filtered_texts

# formating context data
def format_prompt_inputs(user_query, text_collection, image_collection, configuration):
    images = image_uris(image_collection, user_query,
                        max_distance=configuration.image_max_distance_threshold,
                        max_results=configuration.max_retrieved_images)
    text = text_uris(text_collection, user_query,
                     max_distance=configuration.text_max_distance_threshold,
                     max_results=configuration.max_retrieved_text_documents)

    images = [img_path for img_path in images if os.path.exists(img_path)]
    inputs = {'query': user_query, 'texts': text}
    return images, inputs

# process user response
def user_response(gpt_api_key, users_query, text_collection, image_collection, model, configuration: CacheParameters, process_feedback: callable = None):
    try:
        if process_feedback:
            process_feedback("Initializing GPT model...")
        api_key = gpt_api_key
        gpt = ChatOpenAI(model=model, temperature=configuration.temperature/100, api_key=api_key)

        parser = StrOutputParser()

        if process_feedback:
            process_feedback("Formatting inputs (retrieving images and docs)...")

        images, inputs = format_prompt_inputs(users_query, text_collection, image_collection, configuration)

        if process_feedback:
            process_feedback(
                f"Retrieved {len(inputs['texts'])} docs and {len(images)} images. Preparing LLM prompt...")

        # build user prompt
        # Prepare user message content dynamically
        user_message_content = [{
            "type": "text",
            "text": "User's Query:\n{query}\n\nProvided Documents:\n{texts}"
        }]

        # Decoding for llm input
        num_images_to_decode = min(len(images), Config.max_retrieved_images_to_llm)
        for idx in range(num_images_to_decode):
            with open(images[idx], 'rb') as image_file:
                image_data = image_file.read()
            inputs[f'image_data_{idx}'] = base64.b64encode(image_data).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(images[idx])
            if mime_type is None:
                mime_type = "image/jpeg"
            user_message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{{image_data_{idx}}}"
                }
            })

        # system prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a multimodal retrieval assistant.\n\n"
                 "Your task is to answer the user's query using the provided documents and images.\n"
                 "Each document contains a Title, URL, and main Text.\n\n"
                 "Instructions:\n"
                 "- Carefully read the user's query.\n"
                 "- Review the provided documents (each document starts with Title and URL).\n"
                 "- Use information from documents that are clearly relevant to the user's query.\n"
                 "- If you can not provide answer, respond exactly with: 'I am not sure based on the provided documents.'\n"
                 "- If you use information from a document, explicitly cite its Title and URL as source.\n"
                 "- Use primarily the text to form the answer, and optionally reference the images to enhance it.\n"
                 "- DO NOT hallucinate or invent information.\n\n"),
                ("user", user_message_content)
            ]
        )

        if process_feedback:
            process_feedback("Building the prompt...")

        # formatted_prompt = prompt.invoke(inputs)
        # print("------ FULL PROMPT ------")
        # print(formatted_prompt)
        # print("--------------------------")

        chain = prompt | gpt | parser

        if process_feedback:
            process_feedback("Calling the LLM...")

        response = chain.invoke(inputs)
        return response, images, None
    except Exception as e:
        if process_feedback:
            process_feedback(f"Error: {str(e)}")
        return '', [], f"Failed to generate user response: {str(e)}"
