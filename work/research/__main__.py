"""Run with: python -m work.research (from the repository root)."""
from pathlib import Path
import hashlib,os
from .pipeline import run_development
from .figures import plot_development_results


def main():
    folder=Path(__file__).resolve().parent
    names=['baselines','evaluation','warehouse','count_noise','paired','pipeline','figures']
    sources={name:(folder/(name+'.py')).read_text(encoding='utf-8') for name in names}
    extractor=hashlib.sha256(sources['warehouse'].encode()).hexdigest()
    implementation=hashlib.sha256(''.join(n+'\n'+sources[n] for n in names).encode()).hexdigest()
    os.environ.setdefault('MPLCONFIGDIR',str(Path('work/private_cache/matplotlib').resolve()))
    run=run_development(extractor,implementation,
        synthetic=os.environ.get('FLYRANK_CLEAN_SYNTHETIC')=='1',
        cache_path=os.environ.get('FLYRANK_CLEAN_CACHE'))
    plot_development_results(run)


if __name__=='__main__':main()
