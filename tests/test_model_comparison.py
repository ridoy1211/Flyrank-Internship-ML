"""Behavioral checks for the selection boundary, ranking ties, and independent pipelines."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest

BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('comparison',BASE/'work/model_comparison.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import numpy as np
import pandas as pd

def synthetic():
    rng=np.random.default_rng(19)
    n=4800
    impressions=rng.integers(100,10000,n)
    clicks=rng.integers(1,100,n)
    pos=rng.uniform(1,70,n)
    return pd.DataFrame({
        'client_hash_id':np.repeat([f'c{i:02d}' for i in range(16)],300),
        'content_hash_id':[f'p{i:05d}' for i in range(n)],
        'impressions_prior30':impressions,'clicks_prior30':clicks,
        'ctr_prior30':100*clicks/impressions,'avg_position_prior30':pos,
        'content_type':rng.choice(['a','b'],n),
        'main_intent':rng.choice(['information','commerce'],n),
        m.TARGET:rng.binomial(1,.04+.18*(pos<15),n)})

class SelectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=synthetic()
        cls.result=m.compare_models(cls.frame,n_trees=8)

    def test_primary_metric_over_auc_and_simplicity_tie(self):
        t=pd.DataFrame([{'method':'Random Forest','precision@50':.08,'roc_auc':.9},
                        {'method':'Logistic Regression','precision@50':.24,'roc_auc':.7}])
        self.assertEqual(m.choose_candidate(t),'Logistic Regression')
        t['precision@50']=.08
        self.assertEqual(m.choose_candidate(t),'Logistic Regression')

    def test_held_out_labels_cannot_select_candidate(self):
        altered=self.frame.copy()
        clients=set(self.result['test_frame'].client_hash_id)
        mask=altered.client_hash_id.isin(clients)
        altered.loc[mask,m.TARGET]=1-altered.loc[mask,m.TARGET]
        second=m.compare_models(altered,n_trees=8)
        self.assertEqual(second['selected_method'],self.result['selected_method'])
        pd.testing.assert_frame_equal(second['validation'],self.result['validation'])

    def test_row_permutation_cannot_change_split_or_scores(self):
        second=m.compare_models(self.frame.sample(frac=1,random_state=91),n_trees=8)
        pd.testing.assert_frame_equal(second['test'],self.result['test'])
        pd.testing.assert_frame_equal(second['membership'],self.result['membership'])

    def test_constant_scores_expose_arbitrary_top_k(self):
        labels=np.array([0]*50+[1]*50)
        r=m.rank_metrics(np.zeros(100),labels,ks=(50,))
        self.assertEqual(r['roc_auc'],.5)
        self.assertEqual(r['precision@50'],0)
        self.assertEqual(r['tie_expected_precision@50'],.5)
        self.assertEqual(r['tie_min_precision@50'],0)
        self.assertEqual(r['tie_max_precision@50'],1)

    def test_candidates_do_not_share_preprocessing(self):
        a=m.make_model('Logistic Regression')
        b=m.make_model('Random Forest',n_trees=8)
        self.assertIsNot(a.named_steps['prep'],b.named_steps['prep'])

    def test_every_candidate_uses_same_evaluation_rows(self):
        table=self.result['test']
        self.assertEqual(set(table.method),{'Baseline rule',*m.CANDIDATES})
        self.assertEqual(table.n_eval.nunique(),1)
        self.assertEqual(table.positive_count.nunique(),1)
        self.assertEqual(table.base_rate.nunique(),1)
        assignment=self.result['membership']
        sets=[set(assignment.loc[assignment.split==s,'client_hash_id'])
              for s in ['inner_fit','inner_validation','outer_test']]
        self.assertTrue(all(not(sets[i]&sets[j]) for i in range(3) for j in range(i+1,3)))

    def test_duplicate_page_join_fails(self):
        bad=pd.concat([self.frame,self.frame.iloc[:1]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'not unique'):
            m.compare_models(bad,n_trees=8)

    def test_embedded_notebook_logic_matches_tested_module(self):
        nb=json.loads((BASE/'work/notebooks/fix01_model_comparison.ipynb').read_text(encoding='utf-8'))
        source=''.join(nb['cells'][2]['source'])
        self.assertEqual(source,m.__doc__ and (BASE/'work/model_comparison.py').read_text(encoding='utf-8'))

if __name__=='__main__':
    unittest.main(verbosity=2)
