"""
memory/vector_store.py — Semantic vector memory backed by sqlite-vec.

Architecture overview
─────────────────────
Two tables live in the same SQLite database file and are linked by rowid:

  documents (regular table)
  ─────────────────────────
  rowid    INTEGER  — implicit SQLite rowid, the join key to vec0
  doc_id   TEXT     — a stable, caller-assigned identifier (e.g. "src/auth/router.py")
  content  TEXT     — the actual text content that was embedded
  file_path TEXT    — the path as displayed in search results
  metadata TEXT     — arbitrary JSON for the caller to attach extra context

  vec_index (vec0 virtual table managed by sqlite-vec)
  ────────────────────────────────────────────────────
  rowid      INTEGER — links back to documents.rowid
  embedding  FLOAT[] — the raw float32 vector, dimension set at init time

Why sqlite-vec instead of a dedicated vector DB?
  The master plan explicitly requires zero external services. sqlite-vec is a
  compiled C extension loaded into SQLite, so the entire memory system is a
  single .db file the user can inspect, copy, and version-control alongside
  the project. There is no server to spin up or network call to make.

Embedding dimension
  The dimension is set once at store creation time and cannot change afterwards
  without reinitialising the database. The default of 1536 matches the output
  dimension of text-embedding-3-small (OpenAI / LiteLLM), which is what the
  vector_search native tool uses. If you switch embedding models, recreate the
  store with the correct dimension.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any

import sqlite_vec


# The default embedding dimension matches text-embedding-3-small output size.
DEFAULT_EMBEDDING_DIM = 1536


class VectorStoreError(Exception):
    """Raised when the vector store encounters an unrecoverable state."""
    pass


def _encode_vector(floats: list[float]) -> bytes:
    """
    Pack a Python list of floats into a bytes object using 32-bit IEEE 754
    format. This is the binary representation sqlite-vec expects when you
    INSERT into a vec0 table — passing a Python list directly would raise
    a type error from the extension.
    """
    return struct.pack(f"{len(floats)}f", *floats)


def _decode_vector(blob: bytes) -> list[float]:
    """
    Unpack a sqlite-vec binary blob back into a Python list of floats.
    Used when reading stored embeddings back out for inspection or comparison.
    """
    count = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{count}f", blob))


class VectorStore:
    """
    A semantic vector store backed by a single SQLite database file.

    Typical lifecycle
    ─────────────────
      store = VectorStore(db_path=".synapse/memory.db", embedding_dim=1536)
      store.initialize()           # creates tables on first run, no-op if they exist

      store.add_document(          # add or update a file's embedding
          doc_id="src/auth/router.py",
          content="from fastapi import APIRouter...",
          embedding=[0.12, -0.34, ...],  # 1536 floats from your embedding model
          file_path="src/auth/router.py",
          metadata={"language": "python", "size": 1024},
      )

      results = store.search(query_embedding=[...], top_k=5)
      for r in results:
          print(r["file_path"], r["distance"])
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        """
        Args:
            db_path:       Path to the SQLite database file. Use ":memory:" for
                           ephemeral stores (useful in tests and one-off runs).
            embedding_dim: The dimensionality of the embeddings this store holds.
                           Must match the output dimension of the embedding model
                           you are using throughout the project.
        """
        self._db_path      = str(db_path)
        self._embedding_dim = embedding_dim
        self._conn: sqlite3.Connection | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Open the database connection, load the sqlite-vec extension, and create
        the two tables if they do not already exist.

        This is idempotent — calling initialize() on an existing database is
        safe and will not touch or delete any stored data.
        """
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row  # lets us access columns by name

        # Load the sqlite-vec extension. This must happen before any vec0 table
        # operations. enable_load_extension is disabled again immediately after
        # for security — we do not want arbitrary extension loading to be possible
        # after this point.
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Create the content table if it does not exist.
        # The doc_id column has a UNIQUE constraint so that re-indexing a file
        # that already exists simply replaces it rather than duplicating it.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id    TEXT UNIQUE NOT NULL,
                content   TEXT NOT NULL,
                file_path TEXT NOT NULL,
                metadata  TEXT NOT NULL DEFAULT '{}'
            )
        """)

        # Create the vec0 virtual table. The dimension is baked into the schema
        # at creation time and cannot be changed without dropping and recreating.
        # The rowid here implicitly links to documents.rowid via the INSERT order.
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_index
            USING vec0(embedding float[{self._embedding_dim}])
        """)

        self._conn.commit()

    def close(self) -> None:
        """Close the database connection cleanly."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "VectorStore":
        self.initialize()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Write operations ───────────────────────────────────────────────────

    def add_document(
        self,
        doc_id:    str,
        content:   str,
        embedding: list[float],
        file_path: str,
        metadata:  dict[str, Any] | None = None,
    ) -> None:
        """
        Add a document and its embedding to the store, replacing any existing
        entry with the same doc_id.

        The two-table design requires that we insert into both tables and that
        the rowids stay in sync. We achieve this by:
          1. Inserting into documents (which assigns a rowid).
          2. Using last_insert_rowid() to get that rowid.
          3. Inserting into vec_index with the same rowid explicitly set.

        If the doc_id already exists, we delete the old rows from both tables
        first to keep everything consistent.

        Args:
            doc_id:    Stable unique identifier, typically the relative file path.
            content:   The text content that was passed to the embedding model.
            embedding: The float vector output by the embedding model.
            file_path: Display path shown in search results.
            metadata:  Optional dict of extra context (language, size, last modified, etc.)
        """
        self._assert_initialized()

        if len(embedding) != self._embedding_dim:
            raise VectorStoreError(
                f"Embedding has {len(embedding)} dimensions but this store "
                f"was initialised for {self._embedding_dim} dimensions."
            )

        meta_json = json.dumps(metadata or {})

        # Remove any existing entry with this doc_id before inserting the new one.
        self.delete_document(doc_id)

        # Insert the content row first to get the auto-assigned rowid.
        cursor = self._conn.execute(
            "INSERT INTO documents (doc_id, content, file_path, metadata) VALUES (?, ?, ?, ?)",
            (doc_id, content, file_path, meta_json),
        )
        new_rowid = cursor.lastrowid

        # Insert the embedding into vec_index using the same rowid so that
        # the JOIN in search() correctly pairs each vector with its content.
        self._conn.execute(
            "INSERT INTO vec_index (rowid, embedding) VALUES (?, ?)",
            (new_rowid, _encode_vector(embedding)),
        )

        self._conn.commit()

    def delete_document(self, doc_id: str) -> None:
        """
        Remove a document and its embedding from both tables.
        No-op if the doc_id does not exist.
        """
        self._assert_initialized()

        # Find the rowid so we can delete from vec_index by rowid.
        row = self._conn.execute(
            "SELECT rowid FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()

        if row is not None:
            self._conn.execute("DELETE FROM vec_index  WHERE rowid = ?", (row["rowid"],))
            self._conn.execute("DELETE FROM documents  WHERE doc_id = ?", (doc_id,))
            self._conn.commit()

    # ── Read operations ────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find the top_k most semantically similar documents to query_embedding.

        sqlite-vec's KNN syntax uses a MATCH clause on the vec0 virtual table
        combined with a LIMIT to specify k. The result is ordered by distance
        ascending, meaning the most similar document comes first and distance=0
        would be an exact match.

        Returns a list of dicts, each containing:
          doc_id, content, file_path, metadata (dict), distance (float)
        """
        self._assert_initialized()

        if len(query_embedding) != self._embedding_dim:
            raise VectorStoreError(
                f"Query embedding has {len(query_embedding)} dimensions but this store "
                f"expects {self._embedding_dim}."
            )

        rows = self._conn.execute(
            """
            SELECT
                d.doc_id,
                d.content,
                d.file_path,
                d.metadata,
                v.distance
            FROM vec_index v
            JOIN documents d ON d.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (_encode_vector(query_embedding), top_k),
        ).fetchall()

        return [
            {
                "doc_id":    row["doc_id"],
                "content":   row["content"],
                "file_path": row["file_path"],
                "metadata":  json.loads(row["metadata"]),
                "distance":  row["distance"],
            }
            for row in rows
        ]

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single document by its doc_id without any similarity search.
        Returns None if the doc_id does not exist.
        """
        self._assert_initialized()

        row = self._conn.execute(
            "SELECT doc_id, content, file_path, metadata FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()

        if row is None:
            return None

        return {
            "doc_id":    row["doc_id"],
            "content":   row["content"],
            "file_path": row["file_path"],
            "metadata":  json.loads(row["metadata"]),
        }

    def count(self) -> int:
        """Return the total number of documents currently in the store."""
        self._assert_initialized()
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _assert_initialized(self) -> None:
        if self._conn is None:
            raise VectorStoreError(
                "VectorStore has not been initialised. Call initialize() first, "
                "or use the store as a context manager: 'with VectorStore(...) as store:'"
            )