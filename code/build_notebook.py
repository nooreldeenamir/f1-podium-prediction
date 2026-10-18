"""Build the self-contained lab version and optionally execute it in a fresh kernel."""
from pathlib import Path
import argparse
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]

def build():
    cells = []
    for section in sorted((ROOT / 'code' / 'notebook_sections').glob('*.py')):
        text = section.read_text(encoding='utf-8')
        chunks = re.split(r'^# %%([^\n]*)\n', text, flags=re.M)
        for index in range(1, len(chunks), 2):
            kind, body = chunks[index], chunks[index + 1].rstrip()
            if '[markdown]' in kind:
                body = '\n'.join(line[2:] if line.startswith('# ') else
                                 line[1:] if line.startswith('#') else line for line in body.splitlines())
                cells.append(nbformat.v4.new_markdown_cell(body))
            elif body.strip():
                cells.append(nbformat.v4.new_code_cell(body))
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update({'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                              'language_info': {'name': 'python', 'version': '3.12'}})
    target = ROOT / 'F1_Podium_Prediction.ipynb'
    nbformat.write(notebook, target)
    print(f'Built {target.name}: {len(cells)} cells', flush=True)
    return target, notebook

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    target, notebook = build()
    if args.execute:
        from nbclient import NotebookClient
        import os, time
        # A bundled Python executable may have no matching global kernelspec.
        kernel_dir = ROOT / 'tmp' / 'jupyter' / 'kernels' / 'f1-lab-python'
        kernel_dir.mkdir(parents=True, exist_ok=True)
        import json, sys
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'F1 Lab Python', 'language': 'python'}))
        os.environ['JUPYTER_PATH'] = str(ROOT / 'tmp' / 'jupyter') + os.pathsep + os.environ.get('JUPYTER_PATH', '')
        notebook.metadata.kernelspec.name = 'f1-lab-python'
        def on_start(cell, cell_index, **kwargs):
            if cell.cell_type == 'code':
                print(f'Running cell {cell_index + 1}: {cell.source.splitlines()[0][:100]}', flush=True)
        client = NotebookClient(notebook, timeout=1800, kernel_name='f1-lab-python',
                                resources={'metadata': {'path': str(ROOT)}},
                                on_cell_start=on_start)
        started = time.perf_counter()
        run = {'status': 'RUNNING', 'python': sys.executable, 'fresh_kernel': True}
        try:
            client.execute()
            run['status'] = 'PASS'
        except Exception as error:
            run['status'] = 'FAIL'
            run['error'] = str(error)[-8000:]
            raise
        finally:
            run['elapsed_seconds'] = round(time.perf_counter() - started, 2)
            run['executed_code_cells'] = sum(c.cell_type == 'code' and c.execution_count is not None for c in notebook.cells)
            (ROOT / 'tmp' / 'fresh_run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
            evidence_dir = ROOT / 'output'
            evidence_dir.mkdir(exist_ok=True)
            (evidence_dir / 'execution_run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
            notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
            nbformat.write(notebook, target)
        print('Run All completed successfully; executed outputs saved.', flush=True)

if __name__ == '__main__':
    main()
