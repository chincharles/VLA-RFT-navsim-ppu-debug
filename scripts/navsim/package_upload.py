#!/usr/bin/env python3
"""Create a small, fresh upload directory from Git-tracked migration sources."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory outside the source checkout')
    parser.add_argument('--zip', required=True, help='New ZIP archive path')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    target = Path(args.output).resolve()
    archive = Path(args.zip).resolve()
    if target == source or source in target.parents:
        parser.error('Output must be outside the source checkout')
    if archive == target or target in archive.parents:
        parser.error('ZIP must be outside the output directory')
    if target.exists() or archive.exists():
        parser.error('Choose new output and ZIP paths; existing files are never overwritten')
    listed = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')
    files = {p for p in listed if p}
    files.add(str(Path(__file__).resolve().relative_to(source)))
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    excluded = []
    retained_logs = []
    target.mkdir(parents=True)
    for name in sorted(files):
        relative = Path(name)
        src = source / relative
        if not src.exists():
            raise FileNotFoundError(f'Tracked file missing: {name}')
        if src.is_symlink():
            raise ValueError(f'Review symlink before publication: {name}')
        if name.startswith(('image/', 'train/verl/tests/')) or name == '.gitmodules':
            excluded.append(name)
            continue
        if name.startswith('outputs/'):
            # Retain the small committed validation evidence, without generated checkpoints.
            if name.startswith('outputs/local-checks/') and src.suffix in ('.json', '.log'):
                relative = Path('docs/navsim/local-validation') / relative.relative_to('outputs/local-checks')
                retained_logs.append(str(relative))
            else:
                excluded.append(name)
                continue
        if name == 'README.md':
            relative = Path('docs/navsim/UPSTREAM_README.md')
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    intro = (
        '> 精简上传版：训练、评测、ClearML 代码及服务器 CPFS 配置均保留。'
        '不含旧 Git 历史、论文图片、虚拟环境、运行缓存和原机器人测试模型。'
        '详见 `UPLOAD_CONTENTS.json`。原始快照的完整审计历史保留在本地完整项目中。\n\n'
    )
    (target / 'README.md').write_text(intro + (source / 'README_NAVSIM.md').read_text())
    notice = target / 'docs/navsim/UPSTREAM_README.md'
    notice.write_text('> 原始项目说明，仅作来源记录；其中论文图片未包含在精简上传包中。\n\n' + notice.read_text())
    (target / 'UPLOAD_CONTENTS.json').write_text(json.dumps(dict(
        source_commit=commit,
        packaging='Git-tracked working files plus package_upload.py; no Git object history',
        excluded_tracked_files=excluded,
        validation_logs=retained_logs,
        untracked_runtime_artifacts='Excluded: virtualenvs, downloaded weights, generated checkpoints, caches',
        training_and_evaluation_status='Same implementation as source; no new cloud validation claimed',
    ), ensure_ascii=False, indent=2))
    # Avoid accidentally adding future runtime assets to a fresh Git repository.
    with (target / '.gitignore').open('a') as f:
        f.write('\n# Upload-copy local environments and credentials\n.venv*/\n.env\n.env.*\n!.env.example\nclearml.conf\n')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(target.rglob('*')):
            if path.is_file():
                z.write(path, str(Path(target.name) / path.relative_to(target)))
    paths = [p for p in target.rglob('*') if p.is_file()]
    print(json.dumps(dict(directory=str(target), archive=str(archive), files=len(paths),
                         source_bytes=sum(p.stat().st_size for p in paths),
                         zip_bytes=archive.stat().st_size), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
