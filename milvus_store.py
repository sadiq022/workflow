from pymilvus import MilvusClient, Collection, connections

class MilvusStore:
    def __init__(self, uri, collection_name, dim):
        self.client = MilvusClient(uri)
        self.collection_name = collection_name
        self.dim = dim

        # 🔑 Get the internal connection alias used by MilvusClient
        self.using = self.client._using

        # 🔑 Always bind collection if it exists
        if self.client.has_collection(collection_name):
            self.collection = Collection(
                name=collection_name,
                using=self.using
            )
            self.collection.load()
        else:
            self.collection = None

    def reset_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dim,
            metric_type="COSINE",
            auto_id=True
        )

        # 🔑 Explicitly bind Collection to the same connection
        self.collection = Collection(
            name=self.collection_name,
            using=self.using
        )

    def insert(self, records):
        self.client.insert(
            collection_name=self.collection_name,
            data=records
        )

    def search(self, vector, top_k):
        return self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["pdf_name", "page_number", "chunk_index", "text"]
        )
    
    def fetch_neighbors(self, pdf_name, page_number, window=1, limit=10):
        """
        Fetch nearby chunks from the same PDF using metadata filtering.
        """

        expr = (
            f'pdf_name == "{pdf_name}" '
            f'and page_number >= {page_number - window} '
            f'and page_number <= {page_number + window}'
        )

        results = self.collection.query(
            expr=expr,
            output_fields=["pdf_name", "page_number", "text"],
            limit=limit,
        )

        return results
    
    def fetch_by_pdf_and_page_range(self, pdf_name, page, window=1):
        expr = (
            f'pdf_name == "{pdf_name}" '
            f'and page_number >= {page - window} '
            f'and page_number <= {page + window}'
        )

        return self.collection.query(
            expr=expr,
            output_fields=["pdf_name", "page_number", "chunk_index", "text"],
            limit=20,
        )

    def fetch_all_chunks(self):
        """
        Fetch all chunks from Milvus and return them.
        """
        results = self.collection.query(
            expr="",
            output_fields=["pdf_name", "page_number", "chunk_index", "text"],
            limit=10000,  # Adjust the limit as needed
        )
        return results