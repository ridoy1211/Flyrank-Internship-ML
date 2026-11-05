from pathlib import Path
import copy,json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'work'))
from audit_input_guard import fix05_audit_input_sha,verify_fix05_audit_inputs

class AuditInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.original={'protocol_sha256':'fixed','protocol':{'budget':50},'cohort_audits':[{'n':100}],
            'folds':[{'origin':'2026-04-01','arms':[{'cohort_sha256':'same',
                'prediction_sha256':{'Random Forest':'old','Logistic Regression':'fixed','Prior CTR':'fixed'},
                'metrics':[{'method':'Random Forest','hits':2,'auc':.55}]}]}],
            'execution':{'source':{'dataset_revision':'fixed'},'memory_settings':{'available_memory_bytes_at_configuration':100,'threads':1}}}
        self.sha=fix05_audit_input_sha(self.original)
    def test_irrelevant_rf_bytes_and_ram_change_allowed(self):
        changed=copy.deepcopy(self.original)
        changed['folds'][0]['arms'][0]['prediction_sha256']['Random Forest']='new'
        changed['execution']['memory_settings']['available_memory_bytes_at_configuration']=200
        verify_fix05_audit_inputs(changed,self.sha)
        self.assertEqual(self.original['folds'][0]['arms'][0]['prediction_sha256']['Random Forest'],'old')
    def test_cohort_change_rejected(self):
        changed=copy.deepcopy(self.original);changed['folds'][0]['arms'][0]['cohort_sha256']='other'
        with self.assertRaises(RuntimeError):verify_fix05_audit_inputs(changed,self.sha)
    def test_rf_metric_change_rejected(self):
        changed=copy.deepcopy(self.original);changed['folds'][0]['arms'][0]['metrics'][0]['hits']=3
        with self.assertRaises(RuntimeError):verify_fix05_audit_inputs(changed,self.sha)
    def test_required_prediction_change_rejected(self):
        for name in ['Logistic Regression','Prior CTR']:
            changed=copy.deepcopy(self.original);changed['folds'][0]['arms'][0]['prediction_sha256'][name]='other'
            with self.assertRaises(RuntimeError):verify_fix05_audit_inputs(changed,self.sha)
    def test_protocol_and_source_changes_rejected(self):
        for section,key in [('protocol','budget'),('execution','source')]:
            changed=copy.deepcopy(self.original);changed[section][key]='other'
            with self.assertRaises(RuntimeError):verify_fix05_audit_inputs(changed,self.sha)
    def test_dictionary_order_does_not_matter(self):
        changed=json.loads(json.dumps(self.original,sort_keys=True))
        self.assertEqual(fix05_audit_input_sha(changed),self.sha)

if __name__=='__main__':unittest.main(verbosity=2)
