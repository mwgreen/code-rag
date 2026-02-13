"""
AST-based code chunking using tree-sitter (trusted, mature library).
Extracts semantic units (classes, methods, functions) respecting syntax boundaries.
"""

from tree_sitter import Language, Parser
import tree_sitter_java
import tree_sitter_python
import tree_sitter_typescript
from typing import List, Dict
from pathlib import Path


# Parser instances (cached)
_parsers = {}


def get_parser(language: str) -> Parser:
    """Get or create parser for language."""
    if language in _parsers:
        return _parsers[language]

    # Create Language object (with correct function names)
    if language == 'java':
        lang = Language(tree_sitter_java.language())
    elif language == 'python':
        lang = Language(tree_sitter_python.language())
    elif language in ['typescript', 'javascript']:
        lang = Language(tree_sitter_typescript.language_typescript())
    elif language == 'tsx':
        lang = Language(tree_sitter_typescript.language_tsx())
    else:
        return None

    # Create parser with language (pass to constructor)
    parser = Parser(lang)

    _parsers[language] = parser
    return parser


def extract_node_text(source_bytes: bytes, node) -> str:
    """Extract text for a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode('utf-8')


def get_line_number(source_bytes: bytes, byte_offset: int) -> int:
    """Convert byte offset to line number."""
    return source_bytes[:byte_offset].count(b'\n') + 1


def chunk_with_ast(content: str, language: str, path: str, max_size: int = 2000) -> List[Dict]:
    """
    Chunk code using AST parsing to respect semantic boundaries.

    Args:
        content: Source code content
        language: Language (java, python, typescript, javascript)
        path: File path
        max_size: Maximum chunk size in characters

    Returns:
        List of chunks with AST-aware boundaries
    """
    parser = get_parser(language)
    if not parser:
        return None  # Fall back to regex chunking

    source_bytes = content.encode('utf-8')
    tree = parser.parse(source_bytes)
    root = node = tree.root_node

    chunks = []

    # Node types to extract as chunks (from code-chunk best practices)
    if language == 'java':
        chunk_types = ['method_declaration', 'class_declaration', 'interface_declaration', 'enum_declaration']
    elif language == 'python':
        chunk_types = ['function_definition', 'class_definition']
    elif language == 'typescript':
        chunk_types = ['function_declaration', 'method_definition', 'class_declaration',
                      'interface_declaration', 'type_alias_declaration', 'enum_declaration']
    elif language == 'javascript':
        chunk_types = ['function_declaration', 'method_definition', 'class_declaration']
    else:
        chunk_types = []

    # Use iterative traversal (not recursive) to avoid stack overflow on deeply nested code
    stack = [root]

    while stack:
        node = stack.pop()

        # Check if this node is a chunkable type
        if node.type in chunk_types:
            text = extract_node_text(source_bytes, node)

            # Only chunk if reasonable size
            if len(text) >= 50:  # Minimum chunk size
                start_line = get_line_number(source_bytes, node.start_byte)
                end_line = get_line_number(source_bytes, node.end_byte)

                chunk = {
                    'content': text,
                    'start_line': start_line,
                    'end_line': end_line,
                    'node_type': node.type
                }

                # If chunk is too large, try to split by child nodes
                if len(text) > max_size:
                    # For large classes, extract methods individually
                    child_chunks = []
                    for child in node.children:
                        if child.type in chunk_types:
                            child_text = extract_node_text(source_bytes, child)
                            if len(child_text) >= 50:
                                child_chunks.append({
                                    'content': child_text,
                                    'start_line': get_line_number(source_bytes, child.start_byte),
                                    'end_line': get_line_number(source_bytes, child.end_byte),
                                    'node_type': child.type
                                })

                    if child_chunks:
                        chunks.extend(child_chunks)
                    else:
                        chunks.append(chunk)  # Keep large chunk if no children
                else:
                    chunks.append(chunk)

        # Add children to stack for traversal
        stack.extend(reversed(node.children))

    # If no chunks found, return None to fall back to regex
    if not chunks:
        return None

    return chunks


def chunk_code_ast(content: str, language: str, path: str) -> List[Dict]:
    """
    Chunk code using AST if possible, fall back to regex if not.

    Returns:
        List of chunks or None if AST parsing not available
    """
    if language not in ['java', 'python', 'typescript', 'javascript']:
        return None  # AST not available for this language

    try:
        chunks = chunk_with_ast(content, language, path)
        return chunks
    except Exception as e:
        print(f"AST chunking failed for {path}: {e}")
        return None  # Fall back to regex
