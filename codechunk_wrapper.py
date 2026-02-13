"""
Python wrapper for code-chunk (Node.js library with advanced AST features).
Provides contextualized chunking with scope, imports, and signatures.

Two modes:
  - Single file: chunk_with_codechunk() — spawns Node.js per call (for MCP tool use)
  - Batch: CodeChunkBatch context manager — one Node.js process for many files
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional


CHUNKER_SCRIPT = str(Path(__file__).parent / "chunker.mjs")
CHUNKER_BATCH_SCRIPT = str(Path(__file__).parent / "chunker_batch.mjs")

SUPPORTED_LANGUAGES = ['java', 'typescript', 'javascript', 'python', 'rust', 'go']


def _parse_chunks(chunks_data: list) -> List[Dict]:
    """Convert code-chunk output to our internal chunk format."""
    chunks = []
    for c in chunks_data:
        chunk = {
            'content': c['contextualized'],
            'start_line': c['start_line'],
            'end_line': c['end_line'],
            'node_type': c.get('node_type', 'chunk'),
        }
        if c.get('scope'):
            chunk['scope'] = c['scope']
        if c.get('signatures'):
            chunk['signatures'] = c['signatures']
        if c.get('imports'):
            chunk['imports'] = c['imports']
        chunks.append(chunk)
    return chunks


def chunk_with_codechunk(content: str, filepath: str, language: str, max_size: int = 2000) -> Optional[List[Dict]]:
    """
    Chunk a single file using code-chunk (spawns Node.js process).
    Use CodeChunkBatch for bulk indexing instead.
    """
    if language not in SUPPORTED_LANGUAGES:
        return None

    try:
        result = subprocess.run(
            ['node', CHUNKER_SCRIPT, filepath, str(max_size)],
            input=content.encode('utf-8'),
            capture_output=True,
            timeout=30
        )

        if result.returncode != 0:
            error = result.stderr.decode('utf-8', errors='ignore')
            if 'Unsupported' not in error:
                print(f"code-chunk error for {filepath}: {error[:100]}")
            return None

        chunks_data = json.loads(result.stdout)
        chunks = _parse_chunks(chunks_data)
        return chunks if chunks else None

    except subprocess.TimeoutExpired:
        print(f"code-chunk timeout for {filepath}")
        return None
    except Exception as e:
        print(f"code-chunk failed for {filepath}: {e}")
        return None


class CodeChunkBatch:
    """
    Long-lived Node.js process for batch chunking via NDJSON.

    Usage:
        with CodeChunkBatch() as chunker:
            chunks = chunker.chunk(content, filepath, language)
    """

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self._proc = None

    def __enter__(self):
        self._proc = subprocess.Popen(
            ['node', CHUNKER_BATCH_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self

    def __exit__(self, *exc):
        if self._proc:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
            self._proc = None

    def chunk(self, content: str, filepath: str, language: str) -> Optional[List[Dict]]:
        """Chunk a single file through the long-lived Node.js process."""
        if language not in SUPPORTED_LANGUAGES:
            return None
        if not self._proc or self._proc.poll() is not None:
            return None

        try:
            request = json.dumps({
                'filepath': filepath,
                'content': content,
                'max_size': self.max_size,
            })
            self._proc.stdin.write((request + '\n').encode('utf-8'))
            self._proc.stdin.flush()

            line = self._proc.stdout.readline()
            if not line:
                return None

            result = json.loads(line)
            if 'error' in result:
                if 'Unsupported' not in result['error']:
                    print(f"code-chunk error for {filepath}: {result['error'][:100]}")
                return None

            chunks = _parse_chunks(result.get('chunks', []))
            return chunks if chunks else None

        except Exception as e:
            print(f"code-chunk batch failed for {filepath}: {e}")
            return None
