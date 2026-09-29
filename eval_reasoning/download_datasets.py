
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modelscope import snapshot_download

root = Path('./datasets')
datasets = {
    # 'aime24': 'evalscope/aime24',
    # 'aime25': 'evalscope/aime25',
    # 'aime26': 'evalscope/aime26',
    # 'math_500': 'AI-ModelScope/MATH-500',
    # 'gsm8k': 'AI-ModelScope/gsm8k',
    'gpqa_diamond': 'AI-ModelScope/gpqa_diamond',
    'live_code_bench': 'evalscope/livecodebench_code_generation_lite_parquet',
    'arxivmath': 'evalscope/arxivmath',
    'ifbench': 'allenai/IFBench_test',
    # 'erqa': 'evalscope/ERQA',
}


def download(item):
    name, repo = item
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo,
        repo_type='dataset',
        local_dir=str(destination),
        max_workers=8,
    )
    return name, path


with ThreadPoolExecutor(max_workers=5) as pool:
    futures = [pool.submit(download, item) for item in datasets.items()]
    for future in as_completed(futures):
        name, path = future.result()
        print(f'{name}: {path}', flush=True)
