from io import BytesIO
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageOps
from scratch.daily_activity import LABELS, CLASSES, decode, decide, adjust_probabilities
from scratch.activity_prepare import SimilarityIndex, dhash
from scratch.activity_train import calibrate, selective_metrics
from activity_posts import recognize_activity, activity_status
from photo_posts import write_photo_post
from models import ModelError
from server import WorkspaceHandler


class DailyActivityTests(unittest.TestCase):
    def test_mirror_near_duplicates_are_group_candidates(self):
        rng=np.random.default_rng(2026)
        image=Image.fromarray(rng.integers(0,256,(80,120,3),dtype=np.uint8))
        hashes=(dhash(image),dhash(ImageOps.mirror(image)));index=SimilarityIndex();index.add(hashes)
        self.assertEqual(index.matches(hashes[::-1]),[0])
        self.assertEqual(index.matches((hashes[0]^15,)),[0])

    def test_other_and_low_confidence_predictions_abstain(self):
        thresholds=[.7]*8
        p=np.array([.9]+[.01]*8);self.assertTrue(decide(p,thresholds)['accepted'])
        self.assertFalse(decide(np.ones(9)/9,thresholds)['accepted'])
        p=np.array([.01]*8+[.92]);self.assertFalse(decide(p,thresholds)['accepted'])
        p=np.array([.55,.45]+[0.]*7);self.assertFalse(decide(p,[.5]*8)['accepted'])

    def test_other_prior_adjustment_is_applied_before_the_suggestion_rule(self):
        p=np.array([.7]+[0.]*7+[.3])
        self.assertTrue(decide(p,[.5]*8)['accepted'])
        self.assertFalse(decide(adjust_probabilities(p,2.),[.5]*8)['accepted'])

    def test_calibration_requires_correct_and_supported_validation_suggestions(self):
        p=np.tile(np.array([.9]+[.0125]*8),(20,1))
        rule=calibrate(p,np.zeros(20,dtype=int))
        self.assertLessEqual(rule['thresholds'][0],.9)
        self.assertTrue(all(t>1 for t in rule['thresholds'][1:]))
        bad=calibrate(p,np.ones(20,dtype=int))
        self.assertTrue(all(t>1 for t in bad['thresholds']))
        score=selective_metrics(p,np.zeros(20,dtype=int),rule)
        self.assertEqual(score['suggestion_precision'],1.)

    def test_preprocess_keeps_whole_composition(self):
        stream=BytesIO();Image.new('RGB',(160,80),(255,0,0)).save(stream,format='PNG')
        x=decode(stream.getvalue())
        self.assertEqual(x.shape,(3,160,160))
        np.testing.assert_array_equal(x[:,0,0],[127,127,127])
        np.testing.assert_array_equal(x[:,80,80],[255,0,0])

    def test_unfinished_model_does_not_infer(self):
        stream=BytesIO();Image.new('RGB',(64,64)).save(stream,format='PNG')
        with patch('activity_posts.release',return_value=None):
            with self.assertRaises(ModelError):recognize_activity(stream.getvalue())

    def test_mismatched_threshold_file_is_not_reported_ready(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'best.pt').write_bytes(b'checkpoint-placeholder')
            rule={'thresholds':[.8]*8,'margin':.15}
            report={'step':200,'labels':list(CLASSES),'accuracy':.7,'macro_f1':.6,'calibration':rule}
            (folder/'calibration.json').write_text(json.dumps({'step':200,**rule}))
            (folder/'test_metrics.json').write_text(json.dumps(report))
            with patch('activity_posts.DIRECTORY',folder):
                self.assertTrue(activity_status()['ready'])
                (folder/'calibration.json').write_text(json.dumps({'step':200,**rule,'thresholds':[.5]*8}))
                self.assertFalse(activity_status()['ready'])

    def test_activity_writing_requires_confirmation_and_does_not_invent_count(self):
        with patch('photo_posts.rewrite_post',return_value={'versions':['a','b','c']}) as writer:
            for activity,confirmed in (('eating',False),('other',True),('barbecue',True)):
                with self.assertRaises(ValueError):write_photo_post([], '', 'concise', True, activity, confirmed)
            writer.assert_not_called()
            r=write_photo_post([], '', 'concise', True, 'hugging', True)
            self.assertEqual(r['confirmed_activity'],'hugging')
            facts=writer.call_args.args[0]
            self.assertIn('拥抱',facts)
            self.assertNotIn('两个人',facts)
            self.assertNotIn('朋友',facts)
            self.assertTrue(writer.call_args.kwargs['photo'])

    def test_activity_binary_route_and_confirmed_writing_contract(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),WorkspaceHandler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        connection=HTTPConnection('127.0.0.1',server.server_port,timeout=5)
        try:
            with patch('server.recognize_activity',return_value={'accepted':False}) as recognize:
                connection.request('POST','/api/photo-activity',b'photo-bytes',{'Content-Type':'image/jpeg'})
                response=connection.getresponse();self.assertEqual(response.status,200);response.read()
                recognize.assert_called_once_with(b'photo-bytes')
            for confirmed,status in ((False,400),(True,200)):
                with patch('photo_posts.rewrite_post',return_value={'versions':['a','b','c']}):
                    connection.request('POST','/api/photo-post',json.dumps({'labels':[],'confirmed':True,'activity':'eating',
                                                                         'activity_confirmed':confirmed}),{'Content-Type':'application/json'})
                    response=connection.getresponse();self.assertEqual(response.status,status);response.read()
        finally:
            connection.close();server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
