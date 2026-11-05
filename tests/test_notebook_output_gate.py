"""Exercise the actual CI gate against templates and completed notebooks."""
from pathlib import Path
import contextlib,io,json,os,tempfile,textwrap,unittest

REPO=Path(__file__).resolve().parents[1]
workflow=(REPO/'.github/workflows/smoke-test.yml').read_text(encoding='utf-8')
gate=textwrap.dedent(workflow.split("          python3 - <<'EOF'\n",1)[1].split('\n          EOF',1)[0])

class NotebookOutputGateTests(unittest.TestCase):
    def check_notebook(self,cells):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'work/notebooks').mkdir(parents=True)
            (root/'submission').mkdir();(root/'submission/paper_url.txt').write_text('https://example.org/paper\n')
            (root/'work/notebooks/example.ipynb').write_text(json.dumps({'cells':cells}))
            before=Path.cwd();status=0
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(io.StringIO()):
                    try:exec(compile(gate,'workflow_notebook_gate','exec'),{'__name__':'__main__'})
                    except SystemExit as ex:status=ex.code
            finally:os.chdir(before)
            return status

    def test_comment_only_starter_is_not_a_missing_run(self):
        cell={'cell_type':'code','source':['# Write code here, not prose.\n']*20,'outputs':[]}
        self.assertEqual(self.check_notebook([cell]*4),0)

    def test_filled_unexecuted_notebook_still_fails(self):
        cell={'cell_type':'code','source':['value = 1\n']*40,'outputs':[]}
        self.assertEqual(self.check_notebook([cell]),1)

    def test_executed_notebook_passes(self):
        cell={'cell_type':'code','source':['value = 1\n']*40,
              'outputs':[{'output_type':'stream','name':'stdout','text':['PASS\n']}]}
        self.assertEqual(self.check_notebook([cell]),0)

    def test_ipython_code_is_not_treated_as_comments(self):
        cell={'cell_type':'code','source':['%time print(1)\n']*30,'outputs':[]}
        self.assertEqual(self.check_notebook([cell]),1)

if __name__=='__main__':unittest.main()
