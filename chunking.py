"""
Code chunking utilities - AST-based when possible, regex fallback.
Uses code-chunk (Node.js) for contextualized chunks, tree-sitter as the second
choice, and regex/size-based splitting as the last resort.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()  # before importing codechunk_wrapper, which reads CODE_RAG_NODE / CODE_RAG_CHUNKER_TIMEOUT at import
import yaml

import codechunk_wrapper
from codechunk_wrapper import chunk_with_codechunk

logger = logging.getLogger("code-rag.chunking")

# Fallback: tree-sitter AST chunking
try:
    from ast_chunking import chunk_code_ast
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False
    chunk_code_ast = None

# Configuration
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "3000"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "50"))

CODECHUNK_LANGUAGES = ('java', 'python', 'typescript', 'javascript', 'rust', 'go')
AST_LANGUAGES = ('java', 'python', 'typescript', 'javascript')


def chunker_status() -> Dict:
    """Which chunker is active, for /health."""
    st = codechunk_wrapper.status()
    if st["available"]:
        active = "code-chunk"
    elif AST_AVAILABLE:
        active = "tree-sitter"
    else:
        active = "regex"
    return {"active": active, **st}


def detect_language(path: str) -> str:
    """Detect language from file extension."""
    ext = Path(path).suffix.lower()

    lang_map = {
        '.java': 'java',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.properties': 'properties',
        '.gradle': 'gradle',
        '.graphql': 'graphql',
        '.graphqls': 'graphql',
        '.proto': 'protobuf',
    }

    return lang_map.get(ext, 'unknown')


def detect_type(path: str, language: str) -> str:
    """Detect if file is code, documentation, or config."""
    if language in ['yaml', 'markdown']:
        return 'documentation'
    elif language in ['json', 'xml', 'properties', 'gradle']:
        return 'config'
    else:
        return 'code'


def _add_metadata(chunks: List[Dict], path: str) -> List[Dict]:
    """Add standard metadata fields to chunks."""
    lang = detect_language(path)
    file_type = detect_type(path, lang)

    for chunk in chunks:
        chunk['path'] = path
        chunk['language'] = lang
        chunk['type'] = file_type

    return chunks


# Full chunking implementations for better code structure awareness


def chunk_yaml(content: str, path: str) -> List[Dict]:
    """Chunk YAML files."""
    # Keep small files whole
    if len(content) < MAX_CHUNK_SIZE:
        return [{
            'content': content,
            'path': path,
            'language': 'yaml',
            'type': 'documentation',
            'start_line': 1,
            'end_line': content.count('\n') + 1
        }]

    # Split by document separator for large files
    chunks = []
    documents = content.split('\n---\n')

    for doc in documents:
        if len(doc.strip()) >= MIN_CHUNK_SIZE:
            chunks.append({
                'content': doc,
                'path': path,
                'language': 'yaml',
                'type': 'documentation',
                'start_line': 1,
                'end_line': doc.count('\n') + 1
            })

    return chunks


def chunk_default(content: str, path: str) -> List[Dict]:
    """Default chunking by character count."""
    chunks = []
    lang = detect_language(path)
    file_type = detect_type(path, lang)

    # Small files - keep whole
    if len(content) <= MAX_CHUNK_SIZE:
        return [{
            'content': content,
            'path': path,
            'language': lang,
            'type': file_type,
            'start_line': 1,
            'end_line': content.count('\n') + 1
        }]

    # Large files - split by size
    lines = content.split('\n')
    current_chunk = []
    current_size = 0
    start_line = 1

    for i, line in enumerate(lines, 1):
        current_chunk.append(line)
        current_size += len(line)

        if current_size >= MAX_CHUNK_SIZE:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'path': path,
                'language': lang,
                'type': file_type,
                'start_line': start_line,
                'end_line': i
            })
            current_chunk = []
            current_size = 0
            start_line = i + 1

    # Add remaining
    if current_chunk:
        chunk_content = '\n'.join(current_chunk)
        if len(chunk_content) >= MIN_CHUNK_SIZE:
            chunks.append({
                'content': chunk_content,
                'path': path,
                'language': lang,
                'type': file_type,
                'start_line': start_line,
                'end_line': len(lines)
            })

    return chunks


def _regex_chunk(content: str, lang: str, path: str) -> List[Dict]:
    if lang == 'java':
        return chunk_java(content, path)
    if lang == 'javascript':
        return chunk_javascript(content, path)
    if lang == 'typescript':
        return chunk_typescript(content, path)
    if lang == 'yaml':
        return chunk_yaml(content, path)
    return chunk_default(content, path)


def chunk_file(path: str) -> List[Dict]:
    """Chunk a file: code-chunk, then tree-sitter, then regex/size fallback.

    Files that are not valid UTF-8 are decoded with replacement characters rather
    than skipped: one Latin-1 comment used to drop a whole file from the index.
    """
    if not path:
        return []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        logger.warning("Error reading %s: %s", path, e)
        return []
    if not content.strip():
        return []

    lang = detect_language(path)
    chunks: Optional[List[Dict]] = None

    if lang in CODECHUNK_LANGUAGES and codechunk_wrapper.available():
        chunks = chunk_with_codechunk(content, path, lang)
    if not chunks and AST_AVAILABLE and lang in AST_LANGUAGES:
        chunks = chunk_code_ast(content, lang, path)
    if not chunks:
        chunks = _regex_chunk(content, lang, path)

    # Ensure all chunks have metadata (some chunkers already add it)
    for chunk in chunks:
        chunk.setdefault('path', path)
        chunk.setdefault('language', lang)
        chunk.setdefault('type', detect_type(path, lang))

    return chunks


def chunk_java(content: str, path: str) -> List[Dict]:
    """Chunk Java code by class and method boundaries."""
    chunks = []
    lines = content.split('\n')

    # Pattern for class/interface/enum declarations
    class_pattern = re.compile(r'^\s*(public|private|protected)?\s*(static\s+)?(class|interface|enum)\s+(\w+)')
    # Pattern for method declarations
    method_pattern = re.compile(r'^\s*(public|private|protected)\s+(static\s+)?[\w<>,\s]+\s+(\w+)\s*\(')

    current_chunk = []
    chunk_start = 0
    current_class = None
    brace_count = 0
    in_class = False

    for i, line in enumerate(lines):
        # Track braces to detect block boundaries
        brace_count += line.count('{') - line.count('}')

        # Detect class/interface/enum
        class_match = class_pattern.search(line)
        if class_match:
            # Save previous chunk if exists
            if current_chunk and len('\n'.join(current_chunk)) >= MIN_CHUNK_SIZE:
                chunks.append({
                    'content': '\n'.join(current_chunk),
                    'start_line': chunk_start + 1,
                    'end_line': i,
                    'class_name': current_class
                })

            current_class = class_match.group(4)
            current_chunk = [line]
            chunk_start = i
            in_class = True
            continue

        # Add line to current chunk
        current_chunk.append(line)

        # If class ended (brace count back to 0), potentially split
        if in_class and brace_count == 0 and current_chunk:
            if len('\n'.join(current_chunk)) >= MAX_CHUNK_SIZE:
                chunks.append({
                    'content': '\n'.join(current_chunk),
                    'start_line': chunk_start + 1,
                    'end_line': i + 1,
                    'class_name': current_class
                })
                current_chunk = []
                chunk_start = i + 1
                in_class = False

    # Add final chunk
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'start_line': chunk_start + 1,
            'end_line': len(lines),
            'class_name': current_class
        })

    # Fallback if no chunks created
    if not chunks:
        return chunk_default(content, path)

    return _add_metadata(chunks, path)


def chunk_javascript(content: str, path: str) -> List[Dict]:
    """Chunk JavaScript/ExtJS code by Ext.define blocks and functions."""
    chunks = []
    lines = content.split('\n')

    # Pattern for Ext.define, Ext.create, Ext.application
    ext_pattern = re.compile(r"Ext\.(define|create|application|override)\s*\(\s*['\"]?([\w.]+)")
    # Pattern for function declarations
    func_pattern = re.compile(r'(function\s+\w+\s*\(|const\s+\w+\s*=\s*(async\s+)?function|const\s+\w+\s*=\s*\([^)]*\)\s*=>)')

    current_chunk = []
    chunk_start = 0
    current_component = None
    brace_count = 0
    paren_count = 0
    in_ext_define = False

    for i, line in enumerate(lines):
        # Track braces and parens for Ext.define blocks
        brace_count += line.count('{') - line.count('}')
        paren_count += line.count('(') - line.count(')')

        # Detect Ext.define
        ext_match = ext_pattern.search(line)
        if ext_match:
            # Save previous chunk
            if current_chunk and len('\n'.join(current_chunk)) >= MIN_CHUNK_SIZE:
                chunks.append({
                    'content': '\n'.join(current_chunk),
                    'start_line': chunk_start + 1,
                    'end_line': i,
                    'component': current_component
                })

            current_component = ext_match.group(2)
            current_chunk = [line]
            chunk_start = i
            in_ext_define = True
            continue

        current_chunk.append(line)

        # If Ext.define block ended, potentially split
        if in_ext_define and paren_count == 0 and brace_count == 0 and current_chunk:
            if len('\n'.join(current_chunk)) >= MAX_CHUNK_SIZE or i == len(lines) - 1:
                chunks.append({
                    'content': '\n'.join(current_chunk),
                    'start_line': chunk_start + 1,
                    'end_line': i + 1,
                    'component': current_component
                })
                current_chunk = []
                chunk_start = i + 1
                in_ext_define = False

    # Add final chunk
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'start_line': chunk_start + 1,
            'end_line': len(lines),
            'component': current_component
        })

    # Fallback if no chunks created
    if not chunks:
        return chunk_default(content, path)

    return _add_metadata(chunks, path)


def chunk_typescript(content: str, path: str) -> List[Dict]:
    """Chunk TypeScript code by interface, type, class, and function declarations."""
    # For TypeScript, use similar logic to JavaScript but also look for interface/type
    chunks = []
    lines = content.split('\n')

    # Patterns for TypeScript constructs
    ts_pattern = re.compile(r'^\s*(export\s+)?(interface|type|class|enum)\s+(\w+)')
    func_pattern = re.compile(r'(function\s+\w+|const\s+\w+\s*=|export\s+(async\s+)?function)')

    current_chunk = []
    chunk_start = 0
    current_name = None

    for i, line in enumerate(lines):
        # Detect TypeScript construct
        ts_match = ts_pattern.search(line)
        if ts_match and current_chunk and len('\n'.join(current_chunk)) >= MIN_CHUNK_SIZE:
            # Save previous chunk
            chunks.append({
                'content': '\n'.join(current_chunk),
                'start_line': chunk_start + 1,
                'end_line': i,
                'component': current_name
            })
            current_chunk = []
            chunk_start = i

        if ts_match:
            current_name = ts_match.group(3)

        current_chunk.append(line)

        # Split if chunk too large
        if len('\n'.join(current_chunk)) >= MAX_CHUNK_SIZE:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'start_line': chunk_start + 1,
                'end_line': i + 1,
                'component': current_name
            })
            current_chunk = []
            chunk_start = i + 1

    # Add final chunk
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'start_line': chunk_start + 1,
            'end_line': len(lines),
            'component': current_name
        })

    if not chunks:
        chunks = chunk_default(content, path)

    return chunks


# NOTE: earlier definitions of chunk_yaml (line ~87) and chunk_default (line ~118)
# are the active ones. Earlier versions of this file had duplicate definitions
# here that shadowed the originals — and the shadows used a line-based size
# (MAX_CHUNK_SIZE // 10 = 300 lines per chunk) instead of characters, so any
# file under 300 lines became a single chunk regardless of byte size. That
# silently degraded semantic search for most markdown docs. Removed.


