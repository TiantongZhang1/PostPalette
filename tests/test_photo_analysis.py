from io import BytesIO
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import threading
import unittest
from unittest.mock import patch
from PIL import Image
from models import ModelError
from photo_analysis import analyze_photo
from photo_posts import write_photo_post
from server import WorkspaceHandler


def sample_image():
    stream=BytesIO();Image.new('RGB',(64,64)).save(stream,format='PNG');return stream.getvalue()


class UnifiedPhotoTests(unittest.TestCase):
    def test_one_upload_runs_all_modules_without_turning_candidates_into_facts(self):
        content=sample_image()
        with patch('photo_analysis.recognize_activity',return_value={'accepted':False}) as activity,\
             patch('photo_analysis.recognize_photo',return_value={'predictions':[]}) as objects,\
             patch('photo_analysis.analyze_people',return_value={'count_estimate':2}) as people:
            result=analyze_photo(content)
        for handler in (activity,objects,people):handler.assert_called_once_with(content)
        self.assertFalse(result['partial']);self.assertEqual(result['people']['count_estimate'],2)
        self.assertNotIn('confirmed',result)

    def test_missing_module_preserves_other_results_and_reports_partial_failure(self):
        with patch('photo_analysis.recognize_activity',return_value={'suggestion':'eating'}),\
             patch('photo_analysis.recognize_photo',return_value={'predictions':[]}),\
             patch('photo_analysis.analyze_people',side_effect=ModelError('人物模型不可用')):
            result=analyze_photo(sample_image())
        self.assertEqual(result['activity']['suggestion'],'eating')
        self.assertIsNotNone(result['objects']);self.assertIsNone(result['people'])
        self.assertTrue(result['partial']);self.assertEqual(result['errors'][0]['module'],'people')

    def test_bad_image_is_rejected_before_any_model(self):
        with patch('photo_analysis.recognize_activity') as model:
            with self.assertRaises(ValueError):analyze_photo(b'not a photo')
            model.assert_not_called()

    def test_manual_background_only_still_requires_confirmation(self):
        with patch('photo_posts.rewrite_post',return_value={'versions':['a','b','c']}) as writer:
            with self.assertRaises(ValueError):write_photo_post([], '两个人坐在椅子上。', 'concise', False)
            writer.assert_not_called()
            result=write_photo_post([], '两个人坐在椅子上。', 'concise', True)
            self.assertEqual(result['confirmed_labels'],[])
            self.assertIn('两个人坐在椅子上',writer.call_args.args[0])
            self.assertTrue(writer.call_args.kwargs['photo'])

    def test_http_combined_route_upload_and_asset(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),WorkspaceHandler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        connection=HTTPConnection('127.0.0.1',server.server_port,timeout=5)
        try:
            content=sample_image()
            with patch('server.analyze_photo',return_value={'activity':None,'objects':None,'people':None,'errors':[]}) as handler:
                connection.request('POST','/api/photo-analyze',content,{'Content-Type':'image/png'})
                response=connection.getresponse();self.assertEqual(response.status,200)
                self.assertIn('activity',json.loads(response.read()));handler.assert_called_once_with(content)
            connection.request('POST','/api/photo-analyze',b'bad',{'Content-Type':'image/png'})
            response=connection.getresponse();self.assertEqual(response.status,400);response.read()
            connection.request('GET','/photo.js');response=connection.getresponse()
            self.assertEqual(response.status,200);self.assertIn(b'/api/photo-analyze',response.read())
        finally:
            connection.close();server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
