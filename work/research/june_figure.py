"""June figure drawn from aggregate JSON only."""
from pathlib import Path
import json


def plot_june_results(run):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    run=Path(run)
    result=json.loads((run/'june_validation.json').read_text(encoding='utf-8'))
    paired=json.loads((run/'paired_queue_audit.json').read_text(encoding='utf-8'))
    arm=next(a for a in result['folds'][0]['arms'] if a['arm']=='known_clients')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    if arm['status']=='scored':
        rows=[r for r in arm['metrics'] if r['minimum_prior_clicks']==1]
        for i,row in enumerate(rows):
            queue=next(q for q in row['queues'] if q['budget']==50)
            if queue.get('reviewed')!=50:continue
            low=100*queue['precision_lower_bound'];high=100*queue['precision_upper_bound']
            axes[0].plot([low,high],[i,i],color='#0072B2',lw=3,marker='|')
        axes[0].set_yticks(range(len(rows)),[r['method'] for r in rows])
        axes[0].invert_yaxis()
        pair=next(r for r in paired['rows'] if r['arm']=='known_clients')
        for i,row in enumerate(pair['pairs']):
            bound=row['full_portfolio']
            if bound['status']=='evaluated':
                axes[1].plot([100*bound['paired_precision_difference_lower'],100*bound['paired_precision_difference_upper']],
                    [i,i],color='#D55E00',lw=3,marker='|')
        axes[1].set_yticks([0,1],['LR minus prior CTR\n(primary)','LR minus visible CTR excess\n(secondary)'])
        axes[1].invert_yaxis()
    else:
        for ax in axes:ax.text(.5,.5,arm['status'],ha='center',transform=ax.transAxes)
    axes[0].set_title('Precision at 50: logical bounds')
    axes[0].set_xlabel('Percent of the fixed review queue')
    axes[1].set_title('Paired precision difference: logical bounds')
    axes[1].set_xlabel('Percentage points; positive favors LR')
    axes[1].axvline(0,color='0.5',linestyle='--',lw=1)
    for ax in axes:ax.grid(axis='x',alpha=.2)
    prefix='SYNTHETIC CHECK: ' if result['execution']['mode']=='synthetic_fixture_only' else ''
    fig.suptitle(prefix+'June 2026: additional retrospective evaluation, known clients')
    for extension in ['png','svg']:fig.savefig(run/('june_summary.'+extension),dpi=180)
    plt.close(fig)
    caption=('Left: precision bounds for the same fixed review budget of 50, retaining pages whose June labels '
        'are unknown. Right: LR-minus-comparator differences with shared pages canceling. These ranges are '
        'logical missing-label bounds, not confidence intervals. Only the known-client LR-minus-prior-CTR '
        'comparison is primary. All client arms, methods, coverage, secondary budgets and support checks are '
        'retained in the JSON files. Previous June exposure is uncertain; this is not a blind test.\n')
    (run/'figure_caption.txt').write_text(caption,encoding='utf-8')
    return run/'june_summary.png'
