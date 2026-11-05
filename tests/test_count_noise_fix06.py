from pathlib import Path
from math import comb
from fractions import Fraction
import json,sys,unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'work'))
import count_noise_fix06 as n

def exact_reference(i0,c0,i1,c1):
    if not 9*i0<=10*i1<=11*i0:return 0.
    total=c0+c1;weights=[]
    for x in range(max(0,total-i0),min(total-1,i1)+1):
        weights.append((x,comb(i1,x)*comb(i0,total-x)))
    return float(Fraction(sum(w for x,w in weights if 37*x<=17*total),sum(w for x,w in weights)))

def frame06():
    return pd.DataFrame({'impressions_prior28':pd.array([100,100,100,100],dtype='Int64'),
        'clicks_prior28':pd.array([1,20,150,1],dtype='Int64'),
        'impressions_outcome':pd.array([100,100,100,pd.NA],dtype='Int64'),
        'clicks_outcome':pd.array([0,17,60,pd.NA],dtype='Int64'),
        'decoupling_signature':pd.array([1,1,1,pd.NA],dtype='Int64')})

class CountNoiseTests(unittest.TestCase):
    def test_matches_independent_exact_enumeration(self):
        rows=[(i0,c0,i1,c1) for i0 in range(1,13) for i1 in range(1,13)
              for c0 in range(1,i0+1) for c1 in range(i1+1)]
        arrays=np.asarray(rows).T
        expected=np.array([exact_reference(*row) for row in rows])
        actual=n.conditional_signature_probability06(*arrays)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_prior_positive_selection_and_total_conditioning(self):
        q=n.conditional_signature_probability06([100,100,100],[1,2,1],[100]*3,[0,0,1])
        self.assertEqual(q[0],1.)
        self.assertAlmostEqual(q[1],q[2])
        self.assertAlmostEqual(q[2],99/299)

    def test_impression_boundary_and_high_rate_support(self):
        q=n.conditional_signature_probability06([100]*5,[20,20,20,20,100],[89,90,110,111,100],[17,17,17,17,100])
        self.assertEqual(q[0],0);self.assertGreater(q[1],0)
        self.assertGreater(q[2],0);self.assertEqual(q[3],0);self.assertEqual(q[4],0)

    def test_large_impression_small_click_precision(self):
        q=n.conditional_signature_probability06([10**12],[1],[10**12],[1])[0]
        self.assertAlmostEqual(q,exact_reference(10**12,1,10**12,1),places=10)

    def test_invalid_counts_and_scope_rejected(self):
        for args in [([100],[0],[100],[1]),([10],[11],[10],[1]),([100],[1.5],[100],[1]),
                     ([100],[1],[100],[float('nan')]),([1000],[101],[1000],[100])]:
            with self.assertRaises(ValueError):n.conditional_signature_probability06(*args)

    def test_missing_outcomes_cap_and_observed_only_bands(self):
        d=frame06();d.loc[2,'impressions_prior28']=1000;d.loc[2,'impressions_outcome']=1000
        before=d.copy(deep=True);result=n.summarize_count_noise06(d)
        pd.testing.assert_frame_equal(d,before)
        self.assertEqual(result['full_arm_observation']['n_unknown'],1)
        self.assertEqual(result['reference_subset']['n_modeled_pages'],2)
        self.assertEqual(result['reference_subset']['n_observed_pages_excluded_total_clicks_gt_200'],1)
        self.assertEqual(result['reference_subset']['observed_signature_count'],2)
        self.assertNotIn('expected',json.dumps(result['prior_click_bands_observed_only']))
        json.dumps(result,allow_nan=False)

    def test_changed_frozen_labels_rejected(self):
        d=frame06().iloc[:2].copy();d.loc[0,'decoupling_signature']=0
        with self.assertRaisesRegex(ValueError,'frozen integer rule'):n.summarize_count_noise06(d)

    def test_empty_reference_is_explicit(self):
        result=n.summarize_count_noise06(frame06().iloc[:0])
        self.assertEqual(result['reference_subset']['n_modeled_pages'],0)
        self.assertIsNone(result['reference_subset']['observed_signature_rate'])

if __name__=='__main__':unittest.main(verbosity=2)
