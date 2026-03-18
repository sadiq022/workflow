from pymilvus import MilvusClient, Collection, connections, FieldSchema, CollectionSchema, DataType
import json
import logging
from config import MILVUS_URI, MILVUS_HOST, MILVUS_PORT

logger = logging.getLogger(__name__)


class MilvusStore:
    def __init__(self, uri=None, collection_name=None, dim=None):
        """Initialize MilvusStore with proper connection configuration."""
        # Allow override of URI, otherwise use config
        if uri is None:
            uri = MILVUS_URI
        if collection_name is None:
            from config import COLLECTION_NAME
            collection_name = COLLECTION_NAME
        if dim is None:
            from config import EMBEDDING_DIM
            dim = EMBEDDING_DIM
            
        self.collection_name = collection_name
        self.dim = dim
        self.uri = uri
        
        # Establish connection with pymilvus connections module
        # This registers the connection for use with Collection API
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
            logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        except Exception as e:
            logger.debug(f"Connection registration: {e}")
            # Connection might already exist, continue anyway
        
        # Also create MilvusClient for compatibility
        self.client = MilvusClient(uri)
        self.using = "default"

        # Always bind collection if it exists
        if self.client.has_collection(collection_name):
            try:
                self.collection = Collection(
                    name=collection_name,
                    using=self.using
                )
                self.collection.load()
                logger.info(f"Loaded existing collection {collection_name}")
            except Exception as e:
                logger.debug(f"Failed to load existing collection: {e}")
                self.collection = None
        else:
            self.collection = None

    def reset_collection(self):
        """Create a new collection with proper schema for all metadata fields."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        # Define schema with all metadata fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="document_number", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="revision", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="document_title", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page_number", dtype=DataType.INT32),
            FieldSchema(name="chunk_index", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="header", dtype=DataType.VARCHAR, max_length=500, is_nullable=True),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            # Store section_path as JSON string for flexibility
            FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=1000, is_nullable=True),
        ]
        
        schema = CollectionSchema(fields=fields, description="PDF chunks with semantic metadata")
        
        # Create collection using pymilvus Collection API
        try:
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=self.using
            )
            # Create index on vector field for faster search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="vector", index_params=index_params)
            logger.info(f"Created collection {self.collection_name} with vector index")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def load_collection(self):
        """Load collection into memory for faster queries."""
        try:
            if self.collection:
                self.collection.load()
                logger.info(f"Loaded collection {self.collection_name}")
        except Exception as e:
            logger.warning(f"Failed to load collection: {e}")

    def insert(self, records):
        """Insert records into Milvus with structured field mapping."""
        if not records:
            return
        
        processed_records = []
        
        for record in records:
            # Ensure section_path is JSON serialized if it's a list
            section_path = record.get("section_path", [])
            if isinstance(section_path, list):
                section_path = json.dumps(section_path)
            
            processed_record = {
                "vector": record.get("vector"),
                "pdf_name": str(record.get("pdf_name", "unknown"))[:255],
                "document_number": str(record.get("document_number", "Unknown"))[:50],
                "revision": str(record.get("revision", "Unknown"))[:50],
                "document_title": str(record.get("document_title", ""))[:255],
                "page_number": int(record.get("page_number", 0)),
                "chunk_index": str(record.get("chunk_index", 0))[:50],
                "text": str(record.get("text", ""))[:65535],
                "label": str(record.get("label", "general"))[:50],
                "header": str(record.get("header", "") or "")[:500],
                "chunk_type": str(record.get("chunk_type", "atomic"))[:50],
                "section_path": str(section_path)[:1000],
            }
            processed_records.append(processed_record)
        
        try:
            self.collection.insert(processed_records)
            # CRITICAL FIX: Flush data to disk to ensure persistence
            self.collection.flush()
            logger.info(f"Inserted and flushed {len(processed_records)} records")
        except Exception as e:
            logger.error(f"Failed to insert records: {e}")
            raise

    def search(self, vector, top_k, filter_expr=None):
        """
        Search for similar chunks by embedding vector using semantic similarity.
        
        Optional filter_expr parameter allows metadata-based filtering.
        Example: filter_expr='document_number == "RRES 90027"'
        
        Returns chunks with enriched metadata including:
        - pdf_name, page_number, chunk_index
        - document_number, revision, document_title
        - text content
        - label (content type classification)
        - header, section_path (document structure)
        - chunk_type (atomic or sliding_window)
        """
        try:
            # Use Collection API for consistent behavior
            search_params = {
                "data": [vector],
                "anns_field": "vector",
                "param": {"metric_type": "COSINE"},
                "limit": top_k,
                "output_fields": [
                    "pdf_name", "page_number", "chunk_index", "text",
                    "document_number", "revision", "document_title",
                    "label", "header", "chunk_type", "section_path"
                ]
            }
            
            # Add filter expression if provided
            if filter_expr:
                search_params["expr"] = filter_expr
            
            results = self.collection.search(**search_params)
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def fetch_neighbors(self, pdf_name, page_number, window=1, limit=10):
        """
        Fetch nearby chunks from the same PDF using metadata filtering.
        """
        try:
            expr = (
                f'pdf_name == "{pdf_name}" '
                f'and page_number >= {page_number - window} '
                f'and page_number <= {page_number + window}'
            )

            results = self.collection.query(
                expr=expr,
                output_fields=["pdf_name", "page_number", "text", "chunk_index", "label"],
                limit=limit,
            )

            return results
        except Exception as e:
            logger.error(f"Fetch neighbors failed: {e}")
            return []
    
    def fetch_by_pdf_and_page_range(self, pdf_name, page, window=1):
        """Fetch chunks within page range from a specific PDF."""
        try:
            expr = (
                f'pdf_name == "{pdf_name}" '
                f'and page_number >= {page - window} '
                f'and page_number <= {page + window}'
            )

            results = self.collection.query(
                expr=expr,
                output_fields=[
                    "pdf_name", "page_number", "chunk_index", "text",
                    "label", "header", "chunk_type", "section_path"
                ],
                limit=20,
            )

            return results
        except Exception as e:
            logger.error(f"Fetch by range failed: {e}")
            return []

    def fetch_all_chunks(self):
        """
        Fetch all chunks from Milvus with enriched metadata.
        """
        try:
            results = self.collection.query(
                expr="",
                output_fields=[
                    "pdf_name", "page_number", "chunk_index", "text",
                    "label", "header", "chunk_type", "section_path"
                ],
                limit=50000,  # Adjust as needed for your dataset
            )
            logger.info(f"Fetched {len(results)} chunks from Milvus")
            return results
        except Exception as e:
            logger.error(f"Fetch all chunks failed: {e}")
            return []