import logging
import os
import time

from src.models import BuildDatabaseAgentState, ExportDirectoryMetdata

logger = logging.getLogger(__name__)


def loggraph(func):
    def wrapper(*args, **kwargs):
        logger.info(f'{'-' * 100}')
        logger.info(f"STARTED '{func.__name__}'")
        logger.info(f'{'-' * 100}')
        
        start_time = time.time()
        output = func(*args, **kwargs)
        end_time = time.time()

        logger.info(f'{'-' * 100}')
        logger.info(f"ENDED '{func.__name__}' IN {end_time - start_time:.0f} SECONDS")
        logger.info(f'{'-' * 100}')
        return output
    return wrapper


def validate_export_metadata(state: BuildDatabaseAgentState):

    # Checking if export directory is created
    if not os.path.exists(state['export_dir']) or not os.path.isdir(state['export_dir']):
        logger.info(f'No export directory found! Skipping metadata check.')
        return
    
    # Checking if metdata file exists
    metadata_json_path = os.path.join(state['export_dir'], 'metadata.json')
    if not os.path.exists(metadata_json_path) or not os.path.isfile(metadata_json_path):
        raise ValueError(f'Metdata file missing! Unable to find "{metadata_json_path}"')
    
    # Checking if exported content already exists
    book_ids = set([book.id for book in state['books']])
    book_dirnames = [dirname for dirname in os.listdir(state['export_dir']) if dirname in book_ids]
    book_dirnames = [dirname for dirname in book_dirnames if os.path.isdir(os.path.join(state['export_dir'], dirname))]
    if not book_dirnames:
        logger.info(f'No exported content found! Skipping metadata check.')
        return
    
    # Loading metadata
    with open(metadata_json_path, 'r') as file:
        try:
            metadata = ExportDirectoryMetdata.model_validate_json(file.read())
        except Exception as e:
            raise ValueError(f'Unable to read corrupted metadata file!', e)

    def error_message(message: str, remedy: str) -> str:
        return f"""
        {message}

        To continue building the database using previously exported content -
        - {remedy}

        To create a fresh database -
        - Either, remove all contents of "{state['export_dir']}" directory
        - Or, use a different 'export_dir' in the state
        """

    # Checking model
    if state['model'] != metadata.model:
        raise ValueError(error_message(
            'Previously exported model does not match the current model!',
            f"Change state's 'model' to '{metadata.model}'",
        ))
    logger.info(f"Metadata 'model' check passed!")

    # Checking chunk size
    if state['chunk_size'] != metadata.chunk_size:
        raise ValueError(error_message(
            'Previously exported chunk size does not match the current chunk size!',
            f"Change state's 'chunk_size' to '{metadata.chunk_size}'",
        ))
    logger.info(f"Metadata 'chunk_size' check passed!")

    # Checking chunk overlap
    if state['chunk_overlap'] != metadata.chunk_overlap:
        raise ValueError(error_message(
            'Previously exported chunk overlap does not match the current chunk overlap!',
            f"Change state's 'chunk_overlap' to '{metadata.chunk_overlap}'",
        ))
    logger.info(f"Metadata 'chunk_overlap' check passed!")

    # Checking books-related information
    book_dict = {book.id: book for book in state['books']}
    for book_id in book_dirnames:
        logger.info(f'[{book_id}] Checking metadata...')

        # Checking if book is in metadata
        if book_id not in metadata.books:
            logger.info(f'[{book_id}] New book! Skipping metadata check.')
            continue

        book_metadata = metadata.books[book_id]
        book = book_dict[book_id]

        # Matching fields
        fields = {
            'name': (book_metadata.name, book.name, False),
            'url': (book_metadata.url, book.url, False),
            'start_page': (book_metadata.start_page, book.start_page, False),
            'end_page': (book_metadata.end_page, book.end_page, False),
            'chunks': (book_metadata.chunks, len(state['book_chunks'][book_id]), True),
        }

        # Checking fields
        for field, tuple in fields.items():
            metadata_field, book_field, transitive = tuple
            if metadata_field != book_field:
                raise ValueError(error_message(
                    f"Previously exported Book[{book_id}]'s {field} does not match the current {field}!",
                    f'Did someone change Book[{book_id}] source file?' if transitive else f"Change Book[{book_id}]'s '{field}' to '{metadata_field}'",
                ))
            
        logger.info(f'[{book_id}] Metadata check passed!')
