import time
from cryptography.fernet import Fernet
import queue
import streamlit as st
import threading
from scripts.AI_bot import load_vdb, user_response
from config import Config, CacheParameters
import copy

# ------ Init Crypto ------
if "secret_key" not in st.session_state:
    st.session_state.secret_key = Fernet.generate_key()

cipher = Fernet(st.session_state.secret_key)

# ------ Init UI ------
st.set_page_config(page_title="Multimodal RAG System", layout="wide")

# ------ Response Progress ------
response_holder_fed = st.empty()
def streamlit_feedback(msg):
    response_holder_fed.info(msg)


# ------ Core Functions ------
def AI_response(api_key: str,
                query: str,
                model: str,
                cache_configuration: CacheParameters,
                status_queue: queue.Queue):
    try:
        process_feedback = lambda msg: status_queue.put(msg)
        if process_feedback:
            process_feedback("Init VDB...")

        text_collection, image_collection = load_vdb()
        text, images, error = user_response(
            gpt_api_key=api_key,
            users_query=query,
            text_collection=text_collection,
            image_collection=image_collection,
            model=model,
            configuration=cache_configuration,
            process_feedback=process_feedback
        )
        return {"text": text, "images": images, "error": error}
    except Exception as e:
        return {"text": None, "images": None, "error": str(e)}


response_holder = {"text": None, "images_uri": None, "error": None}
status_queue = queue.Queue()

def threaded_response(api_key, query, model, cache_configuration):
    try:
        response = AI_response(api_key, query, model, cache_configuration, status_queue)
        response_holder["text"] = response.get("text")
        response_holder["images_uri"] = response.get("images")
        response_holder["error"] = response.get("error")
    except Exception as e:
        response_holder["text"] = None
        response_holder["images_uri"] = None
        response_holder["error"] = str(e)

# ------ Utility Functions ------
def render_current_cache_state(cache_param: CacheParameters):
    if cache_param.response_error:
        st.error(f"Error: {cache_param.response_error}")
    else:
        if cache_param.response_images:
            num_images_to_display = min(len(cache_param.response_images), cache_param.max_retrieved_images)
            for idx in range(num_images_to_display):
                display_image(cache_param.response_images[idx], caption=f"Relevant Image {idx + 1}")
        if cache_param.response_text:
            st.text_area("Text Response", cache_param.response_text, height=300)

def display_image(img, caption):
    st.image(img, caption=caption, use_container_width=True)

def render_query_form(cache_param):
    with st.form("query_form", clear_on_submit=False, enter_to_submit=False):
        query = st.text_input("Enter your Query",
                              key="query_input",
                              max_chars=Config.max_query_length)

        temperature = st.slider("Temperature", 0, 100,
                                step=1, key="temperature_slider")
        model = st.selectbox(
            "Select Model",
            Config.available_LLM_models,
            key="model_selector"
        )
        submit = st.form_submit_button("Send")
    return submit, query, temperature, model


# ------ Init Cache Configs ------
if "current_cache" not in st.session_state:
    st.session_state.current_cache = CacheParameters()

if "saved_caches" not in st.session_state:
    st.session_state.saved_caches = {}

if "new_cache_name" not in st.session_state:
    st.session_state.new_cache_name = ""

# ------ Streamlit UI ------
st.title("Multimodal RAG System")

# ------ Sidebar Save / Load / Delete Cache Configurations ------
st.sidebar.subheader("Saved Configurations")
new_cache_name = st.sidebar.text_input("New Config Name", key="new_cache_name", value='')

# Save current cache
if st.sidebar.button("Save Current Configuration"):
    if new_cache_name:
        if len(st.session_state.saved_caches) >= Config.cache_max_save_count:
            st.sidebar.error(f"Maximum {Config.cache_max_save_count} saved configurations allowed! Please, delete old saved configuration!")
        else:
            st.session_state.saved_caches[new_cache_name] = copy.deepcopy(st.session_state.current_cache)
            st.sidebar.success(f"Configuration '{new_cache_name}' saved!")
            del st.session_state.new_cache_name
            st.rerun()
    else:
        st.sidebar.error("Please enter a name for the configuration!")

# Select and Load config
if st.session_state.saved_caches:
    selected_cache = st.sidebar.selectbox(
        "Load Saved Configuration", list(st.session_state.saved_caches.keys()), key="select_saved_cache"
    )

    if st.sidebar.button("Load Selected Configuration"):
        st.session_state.current_cache = copy.deepcopy(st.session_state.saved_caches[selected_cache])
        st.session_state.query_input = st.session_state.current_cache.user_query
        st.session_state.temperature_slider = st.session_state.current_cache.temperature
        st.session_state.model_selector = st.session_state.current_cache.user_model
        st.success(f"Configuration '{selected_cache}' loaded!")
        st.rerun()

    if st.sidebar.button("Delete Selected Configuration"):
        del st.session_state.saved_caches[selected_cache]
        st.sidebar.success(f"Configuration '{selected_cache}' deleted!")

# Reset to Default
if st.sidebar.button("Reset to Default"):
    st.session_state.current_cache = CacheParameters()
    response_holder["text"] = None
    response_holder["images_uri"] = None
    response_holder["error"] = None
    # Update session state for widgets
    st.session_state.query_input = st.session_state.current_cache.user_query
    st.session_state.temperature_slider = st.session_state.current_cache.temperature
    st.session_state.model_selector = (
        st.session_state.current_cache.user_model
        if st.session_state.current_cache.user_model in Config.available_LLM_models
        else Config.default_llm_model
    )
    st.rerun()


# ------ Sidebar Settings ------
st.sidebar.title("Settings")
api_key = cipher.encrypt(st.sidebar.text_input("API KEY", value=Config.test_api_key, type="password").encode())

st.session_state.current_cache.text_max_distance_threshold = st.sidebar.number_input(
    "Text Max Distance Threshold", min_value=0.0, max_value=Config.max_limit_text_max_distance_threshold,
    value=st.session_state.current_cache.text_max_distance_threshold, step=0.01
)
st.session_state.current_cache.image_max_distance_threshold = st.sidebar.number_input(
    "Image Max Distance Threshold", min_value=0.0, max_value=Config.max_limit_image_max_distance_threshold,
    value=st.session_state.current_cache.image_max_distance_threshold, step=0.01
)
st.session_state.current_cache.max_retrieved_text_documents = st.sidebar.number_input(
    "Max Retrieved Text Documents", min_value=1, max_value=Config.max_limit_max_retrieved_text_documents,
    value=st.session_state.current_cache.max_retrieved_text_documents, step=1
)
st.session_state.current_cache.max_retrieved_images = st.sidebar.number_input(
    "Max Retrieved Images", min_value=1, max_value=Config.max_limit_max_retrieved_images,
    value=st.session_state.current_cache.max_retrieved_images, step=1
)
st.session_state.current_cache.max_retrieved_images_to_llm = st.sidebar.number_input(
    "Max Retrieved Images to LLM", min_value=1, max_value=Config.max_limit_max_retrieved_images_to_llm,
    value=st.session_state.current_cache.max_retrieved_images_to_llm, step=1
)

# ------ Pre-set widget session state values safely before render ------
if "query_input" not in st.session_state:
    st.session_state.query_input = st.session_state.current_cache.user_query

if "temperature_slider" not in st.session_state:
    st.session_state.temperature_slider = st.session_state.current_cache.temperature

if "model_selector" not in st.session_state:
    st.session_state.model_selector = (
        st.session_state.current_cache.user_model
        if st.session_state.current_cache.user_model in Config.available_LLM_models
        else Config.default_llm_model
    )

# ------ Main Form and Processing ------
submit, query, temperature, model = render_query_form(st.session_state.current_cache)

if submit:
    if api_key is None or query is None:
        st.error("API KEY and Query cannot be empty!")
    elif len(query) > Config.max_query_length:
        st.error(f"Query has a limit of {Config.max_query_length} characters!")
    else:
        with st.spinner("Processing... Please wait."):
            status_queue = queue.Queue()
            thread = threading.Thread(target=threaded_response, args=(cipher.decrypt(api_key).decode(), query, model, st.session_state.current_cache))
            thread.start()

            # wait thread end
            while thread.is_alive():
                try:
                    msg = status_queue.get_nowait()
                    if msg:
                        response_holder_fed.info(msg)
                except queue.Empty:
                    pass
                time.sleep(0.1)

            thread.join()

            # Clean process messages
            response_holder_fed.empty()

            # cache profile update
            st.session_state.current_cache.user_query = query
            st.session_state.current_cache.temperature = temperature
            st.session_state.current_cache.user_model = model
            st.session_state.current_cache.response_text = response_holder["text"]
            st.session_state.current_cache.response_images = response_holder["images_uri"]
            st.session_state.current_cache.response_error = response_holder["error"]

        render_current_cache_state(st.session_state.current_cache)
else:
    render_current_cache_state(st.session_state.current_cache)
