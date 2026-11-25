import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from video_ooc.manifest import validate_jobs,smoke_jobs,read_jsonl


def job(**kwargs):
    value={'exp':'INDUCE','sample_id':'demo','item':'demo','variant':'v0','question':'Example?','video_file':None,'prompt_id':'P0','input_mode':'none'};value.update(kwargs);return value

class ManifestTest(unittest.TestCase):
    def test_reference_leak_rejected(self):
        with self.assertRaisesRegex(ValueError,'eight'):validate_jobs([job(label='A')])
    def test_text_does_not_accept_visual_payload(self):
        with self.assertRaisesRegex(ValueError,'null'):validate_jobs([job(video_file='datasets/a.mp4')])
    def test_duplicate_request_rejected_but_variants_distinct(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):validate_jobs([job(),job()])
        self.assertEqual(validate_jobs([job(),job(variant='v1')]),2)
    def test_swap_and_black_cannot_be_text_controls(self):
        for variant in ['v6','v7']:
            with self.assertRaisesRegex(ValueError,'mismatch'):validate_jobs([job(variant=variant)])
    def test_traversal_and_absolute_paths_rejected(self):
        for path in ['../video.mp4','/tmp/video.mp4','C:\\video.mp4']:
            with self.assertRaises(ValueError):validate_jobs([job(input_mode='video',video_file=path)])
    def test_ladder_smoke_keeps_both_end_ladders(self):
        rows=[job(exp='LADDER',sample_id=s,item=s,variant=f'k{k}',input_mode='video',video_file='datasets/demo.mp4') for s in ['first','last'] for k in range(5)]
        validate_jobs(rows);self.assertEqual(len(smoke_jobs(rows)),10)
    def test_public_examples_are_valid_and_synthetic(self):
        path=Path(__file__).resolve().parents[1]/'data/examples/inference_jobs.jsonl';rows=read_jsonl(path);self.assertEqual(validate_jobs(rows),15);self.assertTrue(all(r['sample_id'].startswith('demo_') for r in rows))
if __name__=='__main__':unittest.main()
