import sys
import os
import asyncio
import json
from pathlib import Path
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.database import db_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def load_github_docs(json_file_path: str = "data/dataset/github_docs_for_embedding.json"):

    json_path = Path(json_file_path)
    
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        return False
    
    # Connect to MongoDB
    logger.info("Connecting to MongoDB...")
    if not await db_manager.connect():
        logger.error("Failed to connect to MongoDB")
        return False
    
    try:
        # Load JSON data
        logger.info(f"Loading data from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            github_docs = json.load(f)
        
        logger.info(f"Found {len(github_docs)} documents to load")

        # Load documents automatically
        success_count = 0
        failed_count = 0
        
        for i, doc in enumerate(github_docs, 1):
            try:
                success = await db_manager.add_document(
                    content=doc['content'],
                    source=doc['source'],
                    category=doc.get('category', 'general')
                )
                if success:
                    success_count += 1
                    if success_count % 10 == 0:
                        logger.info(f"Progress: {success_count}/{len(github_docs)} documents loaded")
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error loading document {i}: {e}")
                failed_count += 1
        
        logger.info(f"Loading complete: {success_count} success, {failed_count} failed")
        final_count = await db_manager.get_document_count()
        logger.info(f"Total documents in database: {final_count}")
        
        return success_count > 0

    except Exception as e:
        logger.error(f"Error during loading: {e}")
        return False
    
    finally:
        await db_manager.disconnect()


async def main():

    json_file = "data/dataset/github_docs_for_embedding.json"
    logger.info("Starting GitHub documentation upload...")
    success = await load_github_docs(json_file)
    if success:
        logger.info("GitHub documentation successfully uploaded!")
    else:
        logger.error("Failed to upload GitHub documentation.")


if __name__ == "__main__":
    asyncio.run(main())
