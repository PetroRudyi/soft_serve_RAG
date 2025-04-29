class Config:
    # path and db configurations
    vector_db_path = 'mm_vdb'
    dataset_path = 'dataset'
    parsed_full_data_path = f'{dataset_path}/the_batch_articles_04_28.json'
    image_dataset_path = f'{dataset_path}/images'
    text_dataset_path = f'{dataset_path}/text'

    # test GPT_API key, insert by default
    # ONLY FOR TEST PURPOSES
    test_api_key = ''

    # All model choose (in my case work only gpt-4o-mini and gpt-4.1-mini because free plan)
    available_LLM_models = [
        "gpt-4o-mini",
        "gpt-4o",
        "o1",
        "o3-mini",
        "gpt-4.1",
        "gpt-4.1-mini"]

    default_llm_model = available_LLM_models[0]

    # image filtration keywords
    filtration_keywords = [
        ('a_message_from','-ad',),
        ('course',),
        ('title_no_text', 'the-batch'),
        ('title_a_message_from', 'batch'),
        ('.webp',)
    ]

    # embedding parameters
    embedding_batch_size = 256
    batch_add_to_collection_batch_size = 5461

    # UI user limits
    max_query_length = 1024  # Max query length
    cache_max_save_count = 3 # How much cache can save user

    max_limit_text_max_distance_threshold = 10.0 # max text distance threshold for vdb search
    max_limit_image_max_distance_threshold = 10.0 # max image distance threshold for vdb search
    max_limit_max_retrieved_text_documents = 10 # max retrieved text documents (actual count can be less, depend on distance_threshold)
    max_limit_max_retrieved_images = 10 # max retrieved images (actual count can be less, depend on distance_threshold)
    max_limit_max_retrieved_images_to_llm = 10 # how much llm will get images as input

    # default user parameters
    default_temperature = 10 # model temperature
    text_max_distance_threshold = 1.8  # text distance threshold for vdb search
    image_max_distance_threshold = 1.5  # image distance threshold for vdb search
    max_retrieved_text_documents = 5  # retrieved text documents (actual count can be less, depend on distance_threshold)
    max_retrieved_images = 3  # retrieved images (actual count can be less, depend on distance_threshold)
    max_retrieved_images_to_llm = 2 # how much llm will get images as input

# Config for cashing user
class CacheParameters:

    def __init__(
            self,
            default_temperature=Config.default_temperature,
            default_text_max_distance_threshold=Config.text_max_distance_threshold,
            default_image_max_distance_threshold=Config.image_max_distance_threshold,
            default_max_retrieved_text_documents=Config.max_retrieved_text_documents,
            default_max_retrieved_images=Config.max_retrieved_images,
            default_max_retrieved_images_to_llm=Config.max_retrieved_images_to_llm,
    ):
        self.temperature = default_temperature
        self.text_max_distance_threshold = default_text_max_distance_threshold
        self.image_max_distance_threshold = default_image_max_distance_threshold
        self.max_retrieved_text_documents = default_max_retrieved_text_documents
        self.max_retrieved_images = default_max_retrieved_images
        self.max_retrieved_images_to_llm = default_max_retrieved_images_to_llm

        self.user_query = None # user query
        self.user_model = None # user model
        self.response_text = None # user model output
        self.response_images = None # user model output (images)
        self.response_error = None # user model output (error)
