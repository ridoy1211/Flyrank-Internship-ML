from pathlib import Path
import importlib.util,unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('baselines03',ROOT/'work/baseline_fix03.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def frame(impressions=(100,1000),clicks=(10,10),positions=(2.,2.)):
    d=pd.DataFrame({'impressions_prior28':impressions,'clicks_prior28':clicks,
                    'avg_position_prior28':positions})
    d['ctr_prior28']=100*d.clicks_prior28/d.impressions_prior28
    return d

class TransparentBaselineChecks(unittest.TestCase):
    def test_pooled_rate_is_impression_weighted(self):
        ref=m.fit_pooled_reference(frame())
        self.assertAlmostEqual(ref['rates'].dropna().iloc[0],100*20/1100)
        self.assertAlmostEqual(ref['global_ctr'],100*20/1100)

    def test_labels_do_not_change_reference_or_scores(self):
        d=frame();d['decoupling_signature']=[0,1]
        scores1,_=m.score_transparent_baselines(m.fit_pooled_reference(d),d)
        d['decoupling_signature']=[1,0]
        scores2,_=m.score_transparent_baselines(m.fit_pooled_reference(d),d)
        for key in scores1:np.testing.assert_array_equal(scores1[key],scores2[key])

    def test_unseen_bucket_uses_training_global_rate(self):
        ref=m.fit_pooled_reference(frame())
        evaluation=frame(impressions=(1000,1000),clicks=(10,10),positions=(60.,60.))
        scores,audit=m.score_transparent_baselines(ref,evaluation)
        self.assertEqual(audit['n_unseen_bucket_global_fallback'],2)
        np.testing.assert_allclose(scores['Visible pooled-CTR shortfall'],100*20/1100-1)

    def test_shortfall_and_excess_are_separate_hypotheses(self):
        ref=m.fit_pooled_reference(frame())
        evaluation=frame(impressions=(1000,1000),clicks=(10,100))
        scores,_=m.score_transparent_baselines(ref,evaluation)
        self.assertGreater(scores['Visible pooled-CTR shortfall'][0],0)
        self.assertEqual(scores['Visible pooled-CTR shortfall'][1],0)
        self.assertEqual(scores['Visible pooled-CTR excess'][0],0)
        self.assertGreater(scores['Visible pooled-CTR excess'][1],0)

    def test_train_row_order_and_evaluation_values_cannot_refit_reference(self):
        d=frame();a=m.fit_pooled_reference(d);b=m.fit_pooled_reference(d.iloc[::-1])
        pd.testing.assert_series_equal(a['rates'],b['rates'])
        before=a['rates'].copy()
        m.score_transparent_baselines(a,frame(impressions=(9000,9000),clicks=(3000,2000)))
        pd.testing.assert_series_equal(a['rates'],before)

    def test_invalid_data_are_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError,'Clicks exceed'):
            m.fit_pooled_reference(frame(impressions=(1,1)))
        d=frame();d.loc[0,'ctr_prior28']=.99
        with self.assertRaisesRegex(ValueError,'CTR does not match'):
            m.fit_pooled_reference(d)

    def test_nullable_warehouse_numeric_types(self):
        d=frame().astype({'impressions_prior28':'Int64','clicks_prior28':'Int64',
                          'ctr_prior28':'Float64','avg_position_prior28':'Float64'})
        scores,_=m.score_transparent_baselines(m.fit_pooled_reference(d),d)
        for score in scores.values():
            self.assertEqual(score.dtype,np.dtype(float))
            self.assertTrue(np.isfinite(score).all())

    def test_requires_reviewed_stage2_state(self):
        with self.assertRaisesRegex(ValueError,'Fix 02'):
            m.evaluate_fixed_stage2({'manifest':{}},None)
        with self.assertRaisesRegex(ValueError,'cohort differs'):
            m.evaluate_fixed_stage2({'manifest':{'status':'exploratory_stage_2_measurement_repair','cohort_sha256':'different'}},None)

    def test_queue_profile_reports_concentration_without_client_ids(self):
        d=frame();d['client_hash_id']=['private_a','private_b'];d['decoupling_signature']=[1,0]
        profile=m.queue_profile(d,[1.,0.],k=2)
        self.assertEqual(profile['largest_client_fraction'],.5)
        self.assertEqual(profile['n_clients_with_positive_hits'],1)
        self.assertNotIn('private_a',str(profile))
        self.assertNotIn('private_b',str(profile))

    def test_single_cell_contains_the_tested_helper_without_future_imports(self):
        s=(ROOT/'work/baseline_fix03_continuation.py').read_text(encoding='utf-8')
        helper=(ROOT/'work/baseline_fix03.py').read_text(encoding='utf-8')
        self.assertTrue(s.startswith(helper))
        self.assertNotIn('from __future__',s)
        compile(s,'continuation','exec')

if __name__=='__main__':unittest.main(verbosity=2)
