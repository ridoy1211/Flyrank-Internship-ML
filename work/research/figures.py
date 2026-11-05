"""Publication figures built exclusively from public aggregate result files."""
from pathlib import Path
import json


def plot_development_results(run):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    run=Path(run)
    paired=json.loads((run/'paired_queue_audit.json').read_text())
    noise=json.loads((run/'count_noise_audit.json').read_text())
    manifest=json.loads((run/'run_manifest.json').read_text())
    synthetic=manifest['execution']['mode']=='synthetic_fixture_only'
    title_prefix='SYNTHETIC CHECK — ' if synthetic else ''
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    colors=['#0072B2','#D55E00']
    origins=paired['origins']
    for j,label in enumerate(['LR minus prior CTR','LR minus visible CTR excess']):
        for i,origin in enumerate(origins):
            bounds=origin['pairs'][j]['full_portfolio']
            if bounds['status']!='evaluated':continue
            x=i+(j-.5)*.16
            lo=100*bounds['paired_precision_difference_lower']
            hi=100*bounds['paired_precision_difference_upper']
            axes[0].plot([x,x],[lo,hi],color=colors[j],lw=3,
                         label=label if i==0 else None)
            axes[0].plot([x,x],[lo,hi],linestyle='',marker='_',color=colors[j],ms=8)
    axes[0].axhline(0,color='0.5',lw=1,linestyle='--')
    axes[0].set_xticks(range(len(origins)),[o['origin'][:7] for o in origins])
    axes[0].set_ylabel('Difference in precision at 50 (percentage points)')
    axes[0].set_title('Paired missing-label bounds; known clients')
    axes[0].legend(fontsize=8)
    rows=[r for r in noise['rows'] if r['arm']=='known_clients']
    for field,label,color in [('observed_signature_rate','Observed',colors[0]),
        ('stationary_reference_expected_signature_rate','Stationary reference',colors[1])]:
        axes[1].plot(range(len(rows)),[100*r['reference_subset'][field] for r in rows],
            marker='o',label=label,color=color)
    axes[1].set_xticks(range(len(rows)),[r['origin'][:7] for r in rows])
    axes[1].set_ylabel('Signatures per 100 modeled pages')
    axes[1].set_title('Count-allocation diagnostic; known clients')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle(title_prefix+'Retrospective development analysis')
    for ext in ('png','svg'):fig.savefig(run/('development_summary.'+ext),dpi=180)
    plt.close(fig)
    caption=('Left: logical lower and upper bounds on the difference between two fixed top-50 '
        'queues, allowing either binary label for each exclusive unknown outcome. Shared pages cancel. '
        'These are not confidence intervals. Positive values favor logistic regression. '
        'Right: observed event rates and expected rates under a common-CTR independent-click reference '
        'on the same complete-case pages with combined clicks at most 200. This diagnostic uses future '
        'counts and is not a forecast. Clients and pages repeat across months; do not treat months or '
        'overlapping arms as independent replications. All other methods and arms remain in the JSON results.\n')
    (run/'figure_caption.txt').write_text(caption,encoding='utf-8')
    return run/'development_summary.png'
