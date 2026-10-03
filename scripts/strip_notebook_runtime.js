// Git clean filter: keep notebook source while omitting execution-only state.
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  try {
    const notebook = JSON.parse(input);
    for (const cell of notebook.cells ?? []) {
      if (cell.cell_type === 'code') {
        cell.execution_count = null;
        cell.outputs = [];
      }
      if (cell.metadata) {
        delete cell.metadata.execution;
        delete cell.metadata.ExecuteTime;
      }
    }
    if (notebook.metadata) {
      delete notebook.metadata.widgets;
      if (notebook.metadata.language_info) {
        delete notebook.metadata.language_info.version;
      }
    }
    process.stdout.write(JSON.stringify(notebook, null, 1) + '\n');
  } catch (error) {
    process.stderr.write(`Notebook filter failed: ${error.message}\n`);
    process.exitCode = 1;
  }
});
