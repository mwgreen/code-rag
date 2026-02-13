#!/usr/bin/env node
/**
 * Batch Node.js chunker using NDJSON streaming.
 *
 * Reads NDJSON from stdin: {"filepath": "...", "content": "...", "max_size": 2000}
 * Writes NDJSON to stdout: {"filepath": "...", "chunks": [...]} or {"filepath": "...", "error": "..."}
 *
 * Single process handles all files — no cold-start overhead per file.
 */

import { chunk } from 'code-chunk';
import { createInterface } from 'readline';

const rl = createInterface({ input: process.stdin });

for await (const line of rl) {
    if (!line.trim()) continue;

    let req;
    try {
        req = JSON.parse(line);
    } catch (e) {
        console.log(JSON.stringify({ filepath: null, error: `Invalid JSON: ${e.message}` }));
        continue;
    }

    const { filepath, content, max_size = 2000 } = req;

    try {
        const chunks = await chunk(filepath, content, {
            maxChunkSize: max_size,
            contextMode: 'full',
            siblingDetail: 'signatures',
        });

        const output = chunks.map(c => ({
            content: c.text,
            contextualized: c.contextualizedText,
            start_line: c.lineRange ? c.lineRange.start + 1 : 1,
            end_line: c.lineRange ? c.lineRange.end + 1 : 1,
            node_type: c.context.entities[0]?.type || 'chunk',
            scope: c.context.scope.map(s => s.name).reverse().join(' > '),
            imports: c.context.imports.slice(0, 10).map(i => i.name),
            signatures: c.context.entities.filter(e => e.signature).map(e => e.signature),
        }));

        console.log(JSON.stringify({ filepath, chunks: output }));
    } catch (error) {
        console.log(JSON.stringify({ filepath, error: error.message }));
    }
}
