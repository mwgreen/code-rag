#!/usr/bin/env node
/**
 * Node.js wrapper for code-chunk library (ES module)
 * Usage: node chunker.mjs <filepath> < code.txt
 */

import { chunk } from 'code-chunk';
import fs from 'fs';

async function main() {
    const filepath = process.argv[2];
    const maxSize = parseInt(process.argv[3] || '2000');

    if (!filepath) {
        console.error('Usage: node chunker.mjs <filepath> [maxSize]');
        process.exit(1);
    }

    // Read code from stdin
    let code = '';
    for await (const chunk of process.stdin) {
        code += chunk;
    }

    try {
        const chunks = await chunk(filepath, code, {
            maxChunkSize: maxSize,
            contextMode: 'full',
            siblingDetail: 'signatures',
        });

        // Convert to Python-friendly format
        const output = chunks.map((c, idx) => ({
            content: c.text,
            contextualized: c.contextualizedText,
            start_line: c.lineRange ? c.lineRange.start + 1 : 1,  // Convert 0-based to 1-based
            end_line: c.lineRange ? c.lineRange.end + 1 : 1,
            node_type: c.context.entities[0]?.type || 'chunk',
            scope: c.context.scope.map(s => s.name).reverse().join(' > '),
            imports: c.context.imports.slice(0, 10).map(i => i.name),
            signatures: c.context.entities.filter(e => e.signature).map(e => e.signature),
        }));

        console.log(JSON.stringify(output));
    } catch (error) {
        console.error('Chunking error:', error.message);
        process.exit(1);
    }
}

main();
