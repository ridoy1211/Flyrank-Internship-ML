from pathlib import Path
import itertools,sys,unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'work'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paired_queue_fix07 as p
import chronological_fix05 as c
from test_chronological_fix05 import synthetic_aggregate

class PairedQueueTests(unittest.TestCase):
    def test_identical_queues_cancel_all_unknowns(self):
        frame=pd.DataFrame({'decoupling_signature':pd.array([pd.NA]*5,dtype='Int64')})
        result=p.paired_bounds07(frame,[5,4,3,2,1],[5,4,3,2,1],k=3)
        self.assertEqual(result['shared_unknown_count'],3)
        self.assertEqual(result['paired_precision_difference_lower'],0)
        self.assertEqual(result['paired_precision_difference_upper'],0)
        self.assertEqual(result['point_difference_if_exclusive_outcomes_observed'],0)

    def test_bounds_match_exhaustive_unknown_allocations(self):
        frame=pd.DataFrame({'decoupling_signature':pd.array([1,pd.NA,pd.NA,0,pd.NA],dtype='Int64')})
        a=np.array([5,4,3,2,1]);b=np.array([1,4,3,2,5]);k=3
        result=p.paired_bounds07(frame,a,b,k)
        outcomes=[]
        for values in itertools.product([0,1],repeat=3):
            labels=np.array([1,values[0],values[1],0,values[2]])
            outcomes.append((labels[[0,1,2]].sum()-labels[[1,2,4]].sum())/k)
        self.assertEqual(result['paired_precision_difference_lower'],min(outcomes))
        self.assertEqual(result['paired_precision_difference_upper'],max(outcomes))
        self.assertEqual(result['shared_unknown_count'],2)

    def test_swapping_methods_negates_bounds(self):
        frame=pd.DataFrame({'decoupling_signature':pd.array([1,0,pd.NA,1,pd.NA],dtype='Int64')})
        a=[5,4,3,2,1];b=[1,2,3,4,5]
        x=p.paired_bounds07(frame,a,b,k=2);y=p.paired_bounds07(frame,b,a,k=2)
        self.assertEqual(x['paired_precision_difference_lower'],-y['paired_precision_difference_upper'])
        self.assertEqual(x['paired_precision_difference_upper'],-y['paired_precision_difference_lower'])

    def test_stable_ties_and_fixed_budget(self):
        frame=pd.DataFrame({'decoupling_signature':[1,0,0]})
        x=p.paired_bounds07(frame,[0,0,0],[0,0,1],k=1)
        self.assertEqual(x['a_observed_hits'],1)
        self.assertEqual(x['b_observed_hits'],0)
        self.assertEqual(p.paired_bounds07(frame,[0]*3,[0]*3,k=4)['status'],'insufficient_candidates')

    def test_client_deletion_reverses_portfolio_contrast_without_refit(self):
        frame=pd.DataFrame({'client_hash_id':['a','a','b','b','c','c'],'decoupling_signature':[1,1,0,0,1,0]})
        a=[6,5,4,3,2,1];b=[1,2,3,4,5,6]
        aliases={'a':'client_001','b':'client_002','c':'client_003'}
        result=p.client_deletion07(frame,a,b,aliases,k=2)
        self.assertEqual(result[0]['point_difference_if_exclusive_outcomes_observed'],-.5)
        self.assertEqual(result[1]['point_difference_if_exclusive_outcomes_observed'],.5)
        self.assertEqual(result[2]['point_difference_if_exclusive_outcomes_observed'],1.)

    def test_invalid_scores_and_labels_rejected(self):
        frame=pd.DataFrame({'decoupling_signature':[0,1,0]})
        for a in [[1,2],[1,2,float('nan')]]:
            with self.assertRaises(ValueError):p.paired_bounds07(frame,a,[1,2,3],k=2)
        frame.loc[0,'decoupling_signature']=3
        with self.assertRaises(ValueError):p.paired_bounds07(frame,[1,2,3],[3,2,1],k=2)

    def test_replay_is_exact_original_lr_and_rules(self):
        origin=c.TEST_ORIGINS05[0]
        train,_=c.build_episode05(synthetic_aggregate(c.EPISODES05[0],4,30),c.EPISODES05[0])
        train=train.loc[train.decoupling_signature.notna()]
        test,_=c.build_episode05(synthetic_aggregate(origin,2,30),origin)
        original,_=c.score_models05(train,test);replay=p.replay_scores07(train,test)
        self.assertEqual(set(replay),{'Logistic Regression','Prior CTR','Visible pooled-CTR excess'})
        for name,values in replay.items():np.testing.assert_array_equal(values,original[name])

if __name__=='__main__':unittest.main(verbosity=2)
