"""
Code chunking utilities - AST-based when possible, regex fallback.
Uses tree-sitter for semantic chunking of Java, Python, TypeScript, JavaScript.
"""

import os
import re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import yaml

# Try to import code-chunk wrapper (best quality - has context features)
try:
    from codechunk_wrapper import chunk_with_codechunk
    CODECHUNK_AVAILABLE = True
    print("[DEBUG] code-chunk available")
except ImportError as e:
    CODECHUNK_AVAILABLE = False
    chunk_with_codechunk = None
    print(f"[DEBUG] code-chunk not available: {e}")

# Fallback: tree-sitter AST chunking
try:
    from ast_chunking import chunk_code_ast
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False
    chunk_code_ast = None

# Load environment variables
load_dotenv()

# Configuration
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))


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
        '.gradle': 'gradle'
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


def chunk_file(path: str) -> List[Dict]:
    """Chunk a file using AST when possible, regex fallback."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []

    lang = detect_language(path)

    # Try code-chunk first (best - has full context: scope, imports, signatures)
    if CODECHUNK_AVAILABLE and lang in ['java', 'python', 'typescript', 'javascript', 'rust', 'go']:
        chunks = chunk_with_codechunk(content, path, lang)
        if chunks:
            # code-chunk succeeded - use contextualized chunks
            pass  # chunks already set
        # code-chunk failed, try tree-sitter AST
        elif AST_AVAILABLE:
            ast_chunks = chunk_code_ast(content, lang, path)
            if ast_chunks:
                chunks = ast_chunks
            else:
                # Both failed, use regex
                if lang == 'java':
                    chunks = chunk_java(content, path)
                elif lang == 'javascript':
                    chunks = chunk_javascript(content, path)
                elif lang == 'typescript':
                    chunks = chunk_typescript(content, path)
                else:
                    chunks = chunk_default(content, path)
        else:
            # code-chunk failed, no AST available, use regex
            if lang == 'java':
                chunks = chunk_java(content, path)
            elif lang == 'javascript':
                chunks = chunk_javascript(content, path)
            elif lang == 'typescript':
                chunks = chunk_typescript(content, path)
            else:
                chunks = chunk_default(content, path)
    # Try AST-based chunking if code-chunk not available
    elif AST_AVAILABLE and lang in ['java', 'python', 'typescript', 'javascript']:
        ast_chunks = chunk_code_ast(content, lang, path)
        if ast_chunks:
            chunks = ast_chunks
        else:
            # AST failed, fall back to regex
            if lang == 'java':
                chunks = chunk_java(content, path)
            elif lang == 'javascript':
                chunks = chunk_javascript(content, path)
            elif lang == 'typescript':
                chunks = chunk_typescript(content, path)
            else:
                chunks = chunk_default(content, path)
    # For other languages, use language-specific chunker
    elif lang == 'yaml':
        chunks = chunk_yaml(content, path)
    else:
        chunks = chunk_default(content, path)

    # Ensure all chunks have metadata (some chunkers already add it)
    for chunk in chunks:
        if 'path' not in chunk:
            chunk['path'] = path
        if 'language' not in chunk:
            chunk['language'] = lang
        if 'type' not in chunk:
            chunk['type'] = detect_type(path, lang)

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


def chunk_yaml(content: str, path: str) -> List[Dict]:
    """Chunk YAML documentation files intelligently."""
    # For YAML docs, try to parse structure
    # If it's a single document, keep it as one chunk (unless too large)
    # If it has sections, split by section

    if len(content) <= MAX_CHUNK_SIZE:
        # Keep small YAML files as single chunks
        return [{
            'content': content,
            'start_line': 1,
            'end_line': len(content.split('\n'))
        }]

    # For large YAML files, split by top-level keys or document separators
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    chunk_start = 0

    for i, line in enumerate(lines):
        # YAML document separator
        if line.strip() == '---' and current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'start_line': chunk_start + 1,
                'end_line': i
            })
            current_chunk = []
            chunk_start = i

        current_chunk.append(line)

        # Split if too large
        if len('\n'.join(current_chunk)) >= MAX_CHUNK_SIZE:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'start_line': chunk_start + 1,
                'end_line': i + 1
            })
            current_chunk = []
            chunk_start = i + 1

    # Add final chunk
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'start_line': chunk_start + 1,
            'end_line': len(lines)
        })

    return chunks if chunks else chunk_default(content, path)


def chunk_default(content: str, path: str) -> List[Dict]:
    """Default chunker for config files and unknown types."""
    lines = content.split('\n')

    # For small files, return as single chunk
    if len(content) <= MAX_CHUNK_SIZE:
        return [{
            'content': content,
            'start_line': 1,
            'end_line': len(lines)
        }]

    # For large files, split into fixed-size chunks with overlap
    chunks = []
    chunk_size = MAX_CHUNK_SIZE // 10  # Rough lines per chunk
    overlap = 10  # Lines of overlap

    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_lines = lines[start:end]

        chunks.append({
            'content': '\n'.join(chunk_lines),
            'start_line': start + 1,
            'end_line': end
        })

        start = end - overlap if end < len(lines) else end

    return chunks


