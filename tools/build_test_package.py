"""Build a module-only ZIP from this repository, without committing or publishing."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def build_package(repo, output):
    repo = Path(repo).resolve()
    manifest = json.loads((repo / 'module.json').read_text(encoding='utf-8'))
    source = repo / 'modules/captioner/__init__.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    module = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CaptionerModule')
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in module.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in {'version', 'release_stage'}}
    if manifest['id'] != 'captioner' or manifest['type'] != 'module':
        raise ValueError('Expected a Captioner module manifest')
    if values['version'] != manifest['version'] or values.get('release_stage', 'stable') != manifest['channel']:
        raise ValueError('Module class version/channel must match module.json')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"cyberhub_module_captioner_v{manifest['version']}.zip"
    runtime_files = ['modules/captioner/__init__.py', 'modules/captioner/requirements.txt']
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in runtime_files:
            archive.write(repo / name, name)
        archive.writestr('resources/cyberhub-package.json', json.dumps({**manifest, 'schema': 2}, indent=2) + '\n')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        for name in runtime_files:
            if archive.read(name) != (repo / name).read_bytes():
                raise ValueError(f'Packaged source differs: {name}')
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_text(f'{digest}  {destination.name}\n', encoding='utf-8')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Directory for the module ZIP and SHA-256 file')
    args = parser.parse_args()
    print(build_package(Path(__file__).resolve().parents[1], args.output))
